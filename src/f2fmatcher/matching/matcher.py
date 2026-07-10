import os
import pickle
import numpy as np
import math
import itertools
import torch
from tqdm import tqdm
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

from f2fmatcher.matching.cost import compute_cost_matrix
from f2fmatcher.matching.spatial import get_centroid_dict, get_neighbors_ref_by_distance
from f2fmatcher.matching.propagation import (
    update_prediction, local_prediction_task, filter_matched_pairs,
    compute_geo_costs_parallel, estimate_affine_transform,
    filter_single_pair, remove_duplicate_matched_labels,
)
from f2fmatcher.classifier.model import PairClassifier


def match_fibers(
    img1, list_label_1, cp_output_1,
    img2, list_label_2, cp_output_2,
    label2index,
    dir_embedding,
    cls_checkpoint,
    device,
    list_k=(3, 5, 7),
    n_initial_guess=80,
    n_pair_selected=4,
    min_cls_logit_init=0.75,
    distance_neighbors_ref=200,
    max_distance_affine=200,
    max_cost_geo_neighbors_sides=30,
    max_cost_geo_neighbors_angles=0.15,
    min_cls_logit=0.5,
    patience_label=5,
    n_neighbors_validation=3,
    n_processes=60,
    n_try_unannotated=1,
    patience_prediction_neighbors=5,
    use_multiprocessing_for_local_prediction=True,
    save_step_prediction=False,
    dir_save_prediction_output=None,
):
    if save_step_prediction:
        step_prediction = {}

    classifier = PairClassifier().to(device)
    classifier.load_state_dict(torch.load(cls_checkpoint, map_location=device))
    classifier.eval()

    scores, spatial_dist = compute_cost_matrix(
        img1, list_label_1, cp_output_1,
        img2, list_label_2, cp_output_2,
        classifier, dir_embedding, list_k,
    )

    centroids_1 = get_centroid_dict(cp_output_1[1], list_label_1)
    centroids_2 = get_centroid_dict(cp_output_2[1], list_label_2)

    c1_arr = np.array(list(centroids_1.values()))
    c2_arr = np.array(list(centroids_2.values()))
    D1 = cdist(c1_arr, c1_arr, metric="euclidean")
    D2 = cdist(c2_arr, c2_arr, metric="euclidean")

    # --- initial guess ---
    selected_combs = []
    n_try_init = 0
    while len(selected_combs) < 3 and n_try_init < 3:
        n_init = min(n_initial_guess, len(list_label_1), len(list_label_2))
        cost_pseudo = (scores + spatial_dist) * (scores > min_cls_logit_init)
        update_mask = np.ones(scores.shape)
        init_pairs = []

        while len(init_pairs) < n_init:
            idx1, idx2 = np.unravel_index(np.argmax(cost_pseudo), cost_pseudo.shape)
            update_mask[idx1, :] = 0
            update_mask[:, idx2] = 0
            cost_pseudo *= update_mask
            init_pairs.append((list_label_1[idx1], list_label_2[idx2]))

        pseudo_c1 = np.array([centroids_1[l1] for l1, _ in init_pairs])
        pseudo_c2 = np.array([centroids_2[l2] for _, l2 in init_pairs])

        if save_step_prediction:
            step_prediction["1_initial_guess"] = init_pairs

        combos, costs_arr = compute_geo_costs_parallel(n_init, n_pair_selected, n_processes, pseudo_c1, pseudo_c2)
        costs_arr = np.array(costs_arr)
        n_comb = math.comb(n_pair_selected, 3)
        ok_sides = (costs_arr[..., 0] < max_cost_geo_neighbors_sides).sum(axis=-1) == n_comb
        ok_angles = (costs_arr[..., 1] < max_cost_geo_neighbors_angles).sum(axis=-1) == n_comb
        selected = np.where(ok_sides & ok_angles)[0]
        selected_combs = list(set(itertools.chain.from_iterable([combos[i] for i in selected])))

        if len(selected_combs) < 3:
            n_initial_guess += 20
            n_try_init += 1

    if n_try_init >= 3 and len(selected_combs) < 3:
        return [], scores, spatial_dist, [cp_output_1, cp_output_2]

    init_pairs = [init_pairs[i] for i in selected_combs]
    matched_labels, matched_indexes, matched_labels_dict, prediction, prediction_update, matrix_cost = \
        update_prediction(init_pairs, scores, spatial_dist, min_cls_logit, label2index)

    if save_step_prediction:
        step_prediction["2_selected_combs_from_initial_guess"] = matched_labels

    # --- local propagation ---
    dict_neighbors_1 = {
        l1: get_neighbors_ref_by_distance(D1, label2index["img1"][l1], list_label_1, distance_neighbors_ref)[1]
        for l1 in tqdm(list_label_1, desc="Build neighbor dict")
    }
    dict_patience = {l1: 0 for l1 in list_label_1}
    dict_unannot = {l1: dict_neighbors_1[l1].copy() for l1 in list_label_1}

    n_current = len(matched_labels)
    n_new = float("inf")
    step = 1

    while n_new > 0.0025 * min(len(list_label_1), len(list_label_2)):
        filtered = []
        n_patience = 0
        for l1, l2 in matched_labels:
            prev = set(dict_unannot[l1])
            curr = {n for n in dict_neighbors_1[l1] if n not in matched_labels_dict}
            dict_unannot[l1] = list(curr)
            if len(curr) == 0:
                dict_patience[l1] = 0
                continue
            if curr == prev:
                dict_patience[l1] += 1
            if dict_patience[l1] > patience_prediction_neighbors:
                n_patience += 1
            else:
                filtered.append((l1, l2))

        args_list = [
            (pair, list_label_1, list_label_2, label2index, D1, D2,
             distance_neighbors_ref, prediction_update, matrix_cost, matched_indexes,
             matched_labels_dict, centroids_1, centroids_2,
             max_cost_geo_neighbors_sides, max_cost_geo_neighbors_angles, patience_label)
            for pair in filtered
        ]

        if use_multiprocessing_for_local_prediction:
            results = Parallel(n_jobs=n_processes)(
                delayed(local_prediction_task)(a) for a in tqdm(args_list, desc="Local prediction"))
        else:
            results = [local_prediction_task(a) for a in tqdm(args_list, desc="Local prediction")]

        matched_labels = list(set(matched_labels + list(itertools.chain.from_iterable(results))))
        matched_labels = filter_matched_pairs(
            matched_labels, label2index, D1, D2, n_neighbors_validation,
            centroids_1, centroids_2, max_cost_geo_neighbors_sides, max_cost_geo_neighbors_angles,
        )
        matched_labels, matched_indexes, matched_labels_dict, prediction, prediction_update, matrix_cost = \
            update_prediction(matched_labels, scores, spatial_dist, min_cls_logit, label2index)

        n_new = len(matched_labels) - n_current
        n_current = len(matched_labels)
        print(f"   Step {step}: +{n_new} new pairs (total {n_current})")

        if save_step_prediction:
            step_prediction.setdefault("3_local_prediction", []).append(matched_labels)

        step += 1

        # intermediate save
        if save_step_prediction and dir_save_prediction_output is not None:
            import pickle
            with open(f"{dir_save_prediction_output}/paired_labels.pkl", "wb") as f:
                pickle.dump(matched_labels, f)
            print(f"   [intermediate save] step {step}: {len(matched_labels)} pairs")

    # --- fill unannotated ---
    if save_step_prediction:
        step_prediction["4_unannotated_prediction"] = []

    for t in range(n_try_unannotated):
        l1_un = [l for l in list_label_1 if l not in matched_labels_dict]
        l2_un = [l for l in list_label_2 if l not in set(matched_labels_dict.values())]

        if len(l1_un) == 0 or len(l2_un) == 0:
            break

        c_annot_1 = np.array([centroids_1[p[0]] for p in matched_labels])
        c_annot_2 = np.array([centroids_2[p[1]] for p in matched_labels])
        tform = estimate_affine_transform(c_annot_1, c_annot_2)
        c1_pred = tform(np.array([centroids_1[l] for l in l1_un]))

        for i, label1 in enumerate(l1_un):
            dists = cdist([c1_pred[i]], [centroids_2[l] for l in l2_un], metric="euclidean")[0]
            cls_scores = scores[
                np.ix_([label2index["img1"][label1]],
                       [label2index["img2"][l] for l in l2_un])
            ][0]

            candidates = np.where((dists < max_distance_affine) & (cls_scores > min_cls_logit))[0]
            candidates = candidates[np.argsort(-cls_scores[candidates])]

            for ci in candidates:
                label2 = l2_un[ci]
                if filter_single_pair(label1, label2, matched_labels, D1, D2,
                                      centroids_1, centroids_2, label2index,
                                      n_neighbors_validation, max_cost_geo_neighbors_sides,
                                      max_cost_geo_neighbors_angles):
                    matched_labels.append((label1, label2))
                    l2_un.remove(label2)
                    matched_labels = remove_duplicate_matched_labels(matched_labels)
                    if save_step_prediction:
                        step_prediction["4_unannotated_prediction"].append(matched_labels)
                    break

        matched_labels = filter_matched_pairs(
            matched_labels, label2index, D1, D2, n_neighbors_validation,
            centroids_1, centroids_2, max_cost_geo_neighbors_sides, max_cost_geo_neighbors_angles,
        )

    if save_step_prediction:
        with open(f"{dir_save_prediction_output}/step_prediction.pkl", "wb") as f:
            pickle.dump(step_prediction, f)

    return matched_labels, scores, spatial_dist, [cp_output_1, cp_output_2]
