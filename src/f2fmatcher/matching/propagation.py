import numpy as np
import itertools
import math
import multiprocessing
from tqdm import tqdm
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from skimage.transform import AffineTransform

from f2fmatcher.matching.spatial import (
    get_centroid_dict, triangle_geometry, costs_geometry,
    get_neighbors_ref_by_distance,
)


def local_prediction_task(args):
    (matched_label, list_label_1, list_label_2, label2index, D1, D2,
     distance_neighbors_ref, prediction_update, matrix_cost, matched_indexes,
     matched_labels_dict, centroids_1, centroids_2,
     max_cost_geo_neighbors_sides, max_cost_geo_neighbors_angles, patience_label) = args

    label1_ref, label2_ref = matched_label
    id1_ref = label2index["img1"][label1_ref]
    id2_ref = label2index["img2"][label2_ref]

    _, neighbors_label_1 = get_neighbors_ref_by_distance(D1, id1_ref, list_label_1, distance_neighbors_ref)
    _, neighbors_label_2 = get_neighbors_ref_by_distance(D2, id2_ref, list_label_2, distance_neighbors_ref)

    neighbors_id_1 = [label2index["img1"][l] for l in neighbors_label_1]
    neighbors_id_2 = [label2index["img2"][l] for l in neighbors_label_2]

    prediction_update_ref = np.zeros(matrix_cost.shape)
    prediction_update_ref[np.ix_(neighbors_id_1, neighbors_id_2)] = 1
    prediction_update_ref *= prediction_update

    matrix_cost_ref = matrix_cost * prediction_update_ref
    matrix_cost_pseudo = matrix_cost_ref.copy()
    matched_local = []

    while np.max(matrix_cost_ref) > 0:
        max_id1, max_id2 = np.unravel_index(np.argmax(matrix_cost_pseudo), matrix_cost_pseudo.shape)
        label1, label2 = list_label_1[max_id1], list_label_2[max_id2]

        matched_idx_1 = [i[0] for i in matched_indexes]
        dists = D1[max_id1, matched_idx_1]
        k_idx = [matched_idx_1[i] for i in np.argsort(dists)[:3]]
        k_lbl_1 = [list_label_1[i] for i in k_idx]
        k_lbl_2 = [matched_labels_dict[l1] for l1 in k_lbl_1]

        side_costs, angle_costs = [], []
        for ni, nj in [(0, 1), (0, 2), (1, 2)]:
            s1, a1 = triangle_geometry(
                centroids_1[label1], centroids_1[k_lbl_1[ni]], centroids_1[k_lbl_1[nj]])
            s2, a2 = triangle_geometry(
                centroids_2[label2], centroids_2[k_lbl_2[ni]], centroids_2[k_lbl_2[nj]])
            cs, ca = costs_geometry(s1, a1, s2, a2)
            side_costs.append(cs)
            angle_costs.append(ca)

        if np.mean(side_costs) < max_cost_geo_neighbors_sides and np.mean(angle_costs) < max_cost_geo_neighbors_angles:
            matched_local.append((label1, label2))
            prediction_update_ref[max_id1, :] = 0
            prediction_update_ref[:, max_id2] = 0
            matrix_cost_ref *= prediction_update_ref
            matrix_cost_pseudo = matrix_cost_ref.copy()
        else:
            matrix_cost_pseudo[max_id1, max_id2] = 0
            if np.max(matrix_cost_pseudo) == 0:
                matrix_cost_pseudo = matrix_cost_ref.copy()

    return matched_local


def update_prediction(matched_pairs, scores, spatial_dist, min_cls_logit, label2index):
    matched_labels = []
    matched_indexes = []
    matched_labels_dict = {}
    prediction = np.zeros(scores.shape)
    prediction_update = np.ones(scores.shape)

    for label1, label2 in matched_pairs:
        id1 = label2index["img1"][label1]
        id2 = label2index["img2"][label2]
        matched_labels.append((label1, label2))
        matched_indexes.append((id1, id2))
        matched_labels_dict[label1] = label2
        prediction[id1, id2] = 1
        prediction_update[id1, :] = 0
        prediction_update[:, id2] = 0

    matrix_cost = (scores + spatial_dist) * (scores > min_cls_logit) * prediction_update
    return matched_labels, matched_indexes, matched_labels_dict, prediction, prediction_update, matrix_cost


def filter_matched_pairs(matched_pairs, label2index, D1, D2, n_neighbors_validation,
                         centroids_1, centroids_2, max_cost_geo_neighbors_sides, max_cost_geo_neighbors_angles):
    l1_list = [p[0] for p in matched_pairs]
    l2_list = [p[1] for p in matched_pairs]

    dup1 = {l for l in l1_list if l1_list.count(l) > 1}
    dup2 = {l for l in l2_list if l2_list.count(l) > 1}
    matched_pairs = [(a, b) for a, b in matched_pairs if a not in dup1 and b not in dup2]

    match_labels_1 = [p[0] for p in matched_pairs]
    match_labels_2 = [p[1] for p in matched_pairs]
    matched_dict = dict(matched_pairs)
    match_idx_1 = [label2index["img1"][l] for l in match_labels_1]

    filtered = []
    for i, label1 in enumerate(match_labels_1):
        label2 = matched_dict[label1]
        idx1 = label2index["img1"][label1]
        idx2 = label2index["img2"][label2]

        dist_1 = D1[np.ix_([idx1], match_idx_1)].flatten()
        n_idx = np.argsort(dist_1)[1:1 + n_neighbors_validation]
        n_lbl_1 = [match_labels_1[j] for j in n_idx]
        n_lbl_2 = [matched_dict[l] for l in n_lbl_1]

        side_costs, angle_costs = [], []
        for ni, nj in itertools.combinations(range(n_neighbors_validation), 2):
            s1, a1 = triangle_geometry(centroids_1[label1], centroids_1[n_lbl_1[ni]], centroids_1[n_lbl_1[nj]])
            s2, a2 = triangle_geometry(centroids_2[label2], centroids_2[n_lbl_2[ni]], centroids_2[n_lbl_2[nj]])
            cs, ca = costs_geometry(s1, a1, s2, a2)
            side_costs.append(cs)
            angle_costs.append(ca)

        if np.mean(side_costs) < max_cost_geo_neighbors_sides and np.mean(angle_costs) < max_cost_geo_neighbors_angles:
            filtered.append((label1, label2))

    return filtered


def filter_single_pair(label1, label2, matched_labels, D1, D2, centroids_1, centroids_2,
                       label2index, n_neighbors_validation, max_cost_geo_neighbors_sides, max_cost_geo_neighbors_angles):
    idx1 = label2index["img1"][label1]

    match_labels_1 = [p[0] for p in matched_labels]
    matched_dict = dict(matched_labels)
    match_idx_1 = [label2index["img1"][l] for l in match_labels_1]

    dist_1 = D1[np.ix_([idx1], match_idx_1)].flatten()
    n_idx = np.argsort(dist_1)[1:1 + n_neighbors_validation]
    n_lbl_1 = [match_labels_1[j] for j in n_idx]
    n_lbl_2 = [matched_dict[l] for l in n_lbl_1]

    side_costs, angle_costs = [], []
    for ni, nj in itertools.combinations(range(n_neighbors_validation), 2):
        s1, a1 = triangle_geometry(centroids_1[label1], centroids_1[n_lbl_1[ni]], centroids_1[n_lbl_1[nj]])
        s2, a2 = triangle_geometry(centroids_2[label2], centroids_2[n_lbl_2[ni]], centroids_2[n_lbl_2[nj]])
        cs, ca = costs_geometry(s1, a1, s2, a2)
        side_costs.append(cs)
        angle_costs.append(ca)

    return (np.mean(side_costs) < max_cost_geo_neighbors_sides and
            np.mean(angle_costs) < max_cost_geo_neighbors_angles)


def geometry_constraints(triple_idx, pseudo_c1, pseudo_c2):
    i1, i2, i3 = triple_idx
    s1, a1 = triangle_geometry(pseudo_c1[i1], pseudo_c1[i2], pseudo_c1[i3])
    s2, a2 = triangle_geometry(pseudo_c2[i1], pseudo_c2[i2], pseudo_c2[i3])
    return costs_geometry(s1, a1, s2, a2)


def compute_geo_costs_parallel(n_initial_guess, n_pair_selected, n_processes, pseudo_c1, pseudo_c2):
    combos = list(itertools.combinations(range(n_initial_guess), n_pair_selected))
    args_list = [(idx, pseudo_c1, pseudo_c2) for idx in combos]

    with multiprocessing.Pool(processes=n_processes) as pool:
        costs = pool.map(_cost_geo_kROIs, args_list)

    return combos, costs


def _cost_geo_kROIs(args):
    k_idx, pseudo_c1, pseudo_c2 = args
    return [geometry_constraints(t, pseudo_c1, pseudo_c2) for t in itertools.combinations(k_idx, 3)]


def estimate_affine_transform(P_list, Q_list):
    tform = AffineTransform()
    tform.estimate(np.array(P_list), np.array(Q_list))
    return tform


def remove_duplicate_matched_labels(matched_pairs):
    l1 = [p[0] for p in matched_pairs]
    l2 = [p[1] for p in matched_pairs]
    dup1 = {x for x in l1 if l1.count(x) > 1}
    dup2 = {x for x in l2 if l2.count(x) > 1}
    return [(a, b) for a, b in matched_pairs if a not in dup1 and b not in dup2]
