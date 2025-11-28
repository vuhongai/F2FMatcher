from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm
import math
import itertools
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from glob import glob
import random
from torchvision import transforms
import pickle
import itertools

from scipy.stats import wasserstein_distance
from joblib import Parallel, delayed
import multiprocessing

from utils_CP_flows import *
from FM_models import *
from FM_train_classifier import *
from FM_dataset import *

def spatial_signature(cp_output, list_labels, k = 5):
    """
    Extract local spatial signature of each ROIs by the distance of k nearest neighbors
    Inputs:
    - cp_output : outputs from Cellpose
    - list_labels: list of label_ids of interest
    - k: numbers of neighbors
    Outputs:
    - neighbors_dict: dictionary of sorted k nearest neighbors (label_id)
    - distance_matrix (len(list_labels), k): distance of k nearest neighbors to ROIs
    """
    masks, props, img_rec = cp_output
    centroids = np.array([region.centroid for region in props])
    all_labels = np.array([region.label for region in props])
    label_to_index = {label: i for i, label in enumerate(all_labels)}
    D = cdist(centroids, centroids, metric='euclidean')
    
    distance_matrix = np.zeros((len(list_labels), k), dtype=np.float32)
    neighbors_dict = {}
    
    for i,label in enumerate(list_labels):
        idx = label_to_index[label]
        distances = D[idx]
    
        # Exclude self
        distances[idx] = np.inf
    
        # Get k nearest indices (sorted)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [all_labels[j] for j in nearest_indices]
        nearest_distances = distances[nearest_indices]
    
        neighbors_dict[label] = nearest_labels
        distance_matrix[i] = nearest_distances
    return neighbors_dict, distance_matrix

def inverse_l2_similarity_matrix(A, B):
    """
    Computes the inverse L2 similarity between all rows of A and all rows of B.
    A: (n, k)
    B: (m, k)
    Returns: (n, m) matrix of similarities
    """
    # Expand and compute pairwise L2 distances
    A_sq = np.sum(A**2, axis=1, keepdims=True)       # (n, 1)
    B_sq = np.sum(B**2, axis=1, keepdims=True).T     # (1, m)
    AB = np.dot(A, B.T)                              # (n, m)

    # L2 distance squared = ||a - b||² = ||a||² + ||b||² - 2a·b
    dists_squared = A_sq + B_sq - 2 * AB
    dists = np.sqrt(np.maximum(dists_squared, 1e-8))  # Avoid sqrt of negatives

    # Inverse similarity: higher similarity = smaller distance
    similarity = 1 / (1 + dists)
    return similarity

def plot_prediction(matrix, list_label_1, list_label_2, annotated_pairs=None, cmap="gray_r"):
    """
    matrix shape: len(list_label_1) x len(list_label_2)
    """
    plt.imshow(matrix, cmap=cmap)
    plt.xticks(ticks=np.arange(len(list_label_2)), labels=list_label_2)
    plt.yticks(ticks=np.arange(len(list_label_1)), labels=list_label_1)
    if annotated_pairs != None:
        annotated_indices = [
            (list_label_1.index(p1), list_label_2.index(p2))
            for p1, p2 in annotated_pairs
            if (p1 in list_label_1) and (p2 in list_label_2)
        ]
        ys, xs = zip(*annotated_indices)
        plt.scatter(xs, ys, color='red', s=20)
    plt.show()

# def GetClassifierLogitsFromLatentVector(
#     img1, list_label_1,
#     img2, list_label_2,
#     classifier,
#     dir_embedding,
# ):
#     """
#     Extract the classification logits from all ROI pairs of list_label_1 (of img1) and list_label_2 (of img2).
#     Inputs:
#     - img1, img2: base name of images
#     - list_label_1, list_label_2: lists of label_id for comparison (not necessary all ROIs from in img1/2)
#     - classifier: classifier model
#     - dir_embedding: directory containing all precomputed embedding from VAE model.
#         The completed path should be as followed: f"{dir_embedding}/{img1}_roi_{label1}.npy"
#     Output:
#     - scores: Matrix of shape (len(list_label_1), len(list_label_2))
#     """
#     device = next(classifier.parameters()).device
#     scores = np.zeros((len(list_label_1), len(list_label_2)))
#     with torch.no_grad():
#         for i1,label1 in tqdm(enumerate(list_label_1)):
#             for i2,label2 in enumerate(list_label_2):
#                 emb1 = torch.tensor(np.load(f"{dir_embedding}/{img1}_roi_{label1}.npy"), dtype=torch.float32).to(device)
#                 emb2 = torch.tensor(np.load(f"{dir_embedding}/{img2}_roi_{label2}.npy"), dtype=torch.float32).to(device)
#                 scores[i1,i2] = classifier(emb1, emb2).squeeze().cpu().numpy()
#     return scores

def GetClassifierLogitsFromLatentVector(
    img1, list_label_1,
    img2, list_label_2,
    classifier,
    dir_embedding,
    batch_size=1024
):
    """
    Batched version: compute classification scores between all ROI pairs.
    """
    device = next(classifier.parameters()).device

    # Load embeddings for all ROIs in batch
    emb1_list = [torch.tensor(np.load(os.path.join(dir_embedding, f"{img1}_roi_{label1}.npy")), dtype=torch.float32) for label1 in list_label_1]
    emb2_list = [torch.tensor(np.load(os.path.join(dir_embedding, f"{img2}_roi_{label2}.npy")), dtype=torch.float32) for label2 in list_label_2]

    emb1_tensor = torch.stack(emb1_list).to(device)  # shape (N1, D)
    emb2_tensor = torch.stack(emb2_list).to(device)  # shape (N2, D)

    # Create all pair combinations (i1, i2)
    idx_pairs = list(itertools.product(range(len(list_label_1)), range(len(list_label_2))))
    scores = torch.zeros(len(list_label_1), len(list_label_2), device=device)

    # Run in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(idx_pairs), batch_size)):
            batch_pairs = idx_pairs[i:i+batch_size]
            batch_emb1 = torch.stack([emb1_tensor[i1] for i1, _ in batch_pairs])
            batch_emb2 = torch.stack([emb2_tensor[i2] for _, i2 in batch_pairs])

            logits = classifier(batch_emb1, batch_emb2).squeeze()  # shape (B,)
            for (i1, i2), logit in zip(batch_pairs, logits):
                scores[i1, i2] = logit

    return scores.cpu().numpy()

def wasserstein_similarity_matrix(D1, D2):
    n, m = len(D1), len(D2)
    sim = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist = wasserstein_distance(D1[i], D2[j])
            sim[i, j] = 1 / (1 + dist)
    return sim

def wasserstein_similarity_matrix_parallel(D1, D2, n_jobs=-1):
    n, m = len(D1), len(D2)
    def compute_row(i):
        return [1 / (1 + wasserstein_distance(D1[i], D2[j])) for j in range(m)]
    sim = Parallel(n_jobs=n_jobs)(delayed(compute_row)(i) for i in range(n))
    return np.array(sim)

def GetCostMatrix(
    img1, list_label_1,
    img2, list_label_2,
    classifier,
    dir_embedding,
    images_dir, 
    list_k = [3,5,7],
):
    # get classification logits
    scores = GetClassifierLogitsFromLatentVector(
        img1, list_label_1,
        img2, list_label_2,
        classifier,
        dir_embedding,
    )

    # get spatial signatures
    cp_output_1 = segment_image( f"{images_dir}/{img1}.png")
    cp_output_2 = segment_image( f"{images_dir}/{img2}.png")

    # combine classification logits with spatial signatures
    spatial_dist = np.ones(scores.shape)
    for k in list_k:
        neighbors_dict_1, distance_matrix_1 = spatial_signature(cp_output_1, list_label_1, k)
        neighbors_dict_2, distance_matrix_2 = spatial_signature(cp_output_2, list_label_2, k)
        spatial_dist = spatial_dist * wasserstein_similarity_matrix_parallel(distance_matrix_1, distance_matrix_2)
    spatial_dist = spatial_dist ** (1/len(list_k))
    return scores, spatial_dist, [neighbors_dict_1, neighbors_dict_2], [cp_output_1, cp_output_2]

def get_centroid_dict(props, list_labels):
    return {region.label: np.array(region.centroid) for region in props if region.label in list_labels}

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle(opposite, side1, side2):
    cos_val = (side1**2 + side2**2 - opposite**2) / (2*side1*side2)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_val))
    return angle
    
def triangle_geometry(A, B, C):
    AB = dist(A, B)
    BC = dist(B, C)
    CA = dist(C, A)
    angle_ABC = angle(CA,AB,BC)
    angle_BCA = angle(AB,BC,CA)
    angle_CAB = angle(BC,CA,AB)
    return np.array([AB, BC, CA]), np.array([angle_BCA, angle_CAB, angle_ABC])

def cost_geometry_constraints(sides_1, angles_1, sides_2, angles_2):
    cost = mean_absolute_error(sides_1, sides_2) * mean_absolute_error(angles_1/180, angles_2/180)
    return cost

def get_k_nearest_neighbor(D, idx, k=3):
    distances = D[idx]
    distances[idx] = np.inf
    return np.argsort(distances)[:k]

def get_neighbors_ref(Di, idi_ref, list_label_i, n_neighbors_ref):
    dist_ref_i = Di[idi_ref,:]
    neighbors_ref_id_i = [i for i in np.argsort(dist_ref_i)[:n_neighbors_ref]]
    neighbors_ref_label_i = [list_label_i[i] for i in neighbors_ref_id_i]
    return neighbors_ref_id_i, neighbors_ref_label_i

def local_prediction_task(args):
    matched_label, \
    list_label_1, list_label_2, D1, D2, \
    n_neighbors_ref, \
    prediction_update, matrix_cost, \
    matched_indexes, matched_labels_dict, centroids_1, centroids_2, \
    max_cost_geo_neighbors, patience_label = args

    label1_ref, label2_ref = matched_label
    label2index = {
        "img1": {l:i for i,l in enumerate(list_label_1)},
        "img2": {l:i for i,l in enumerate(list_label_2)},
    }
    id1_ref, id2_ref = label2index['img1'][label1_ref], label2index['img2'][label2_ref]

    neighbors_ref_id_1, neighbors_ref_label_1 = get_neighbors_ref(D1, id1_ref, list_label_1, n_neighbors_ref)
    neighbors_ref_id_2, neighbors_ref_label_2 = get_neighbors_ref(D2, id2_ref, list_label_2, n_neighbors_ref)

    ## update the prediction_update_ref (local prediction)
    # only care about pairs between neighbors of selected pair
    prediction_update_ref = np.zeros(matrix_cost.shape)
    prediction_update_ref[np.ix_(neighbors_ref_id_1, neighbors_ref_id_2)] = 1
    # remove the annotated pairs in these neighbors
    prediction_update_ref = prediction_update * prediction_update_ref

    # only care about pairs between annatated neighbors of selected pair
    matrix_cost_ref = matrix_cost * prediction_update_ref

    # keep track of how many times the pair is evaluated but didn't pass the filters
    matrix_patience_ref = np.zeros(matrix_cost.shape)

    # initiate temporary matrix_cost_pseudo as the matrix_cost_ref of the neighbors
    # at each step, the i-j pair with highest cost will be selected to evaluated, 
        # if the selected i-j pair pass, 
        #    update the local [matrix_cost_ref, prediction_update_ref] 
        #    reinitiate the matrix_cost_pseudo = matrix_cost_ref.copy()
        # if the selected i-j pair did not pass the filter, 
        #    label matrix_cost_pseudo[i,j] = 0
        #    move the the next pair with highest score
    
    matrix_cost_pseudo = matrix_cost_ref.copy()
    matched_labels_local = []

    while np.max(matrix_cost_ref)>0:
        max_id1, max_id2 = np.unravel_index(np.argmax(matrix_cost_pseudo), matrix_cost_pseudo.shape)
        label1, label2 = list_label_1[max_id1], list_label_2[max_id2]
    
        # select 3 ROI1s (from img1) in  closest to label1
        matched_indexes_img1 = [i[0] for i in matched_indexes]
        dist_maxid1 = D1[max_id1, matched_indexes_img1]
        k_neighbors_idx_1 = [matched_indexes_img1[i] for i in np.argsort(dist_maxid1)[:3]]
        k_neighbors_labels_1 = [list_label_1[i] for i in k_neighbors_idx_1]
        k_neighbors_labels_2 = [matched_labels_dict[l1] for l1 in k_neighbors_labels_1]
    
        costs_geo_neighbors = []
        for n_i, n_j in [(0,1), (0,2), (1,2)]:
            sides_1, angles_1 = triangle_geometry(centroids_1[label1], centroids_1[k_neighbors_labels_1[n_i]], centroids_1[k_neighbors_labels_1[n_j]])
            sides_2, angles_2 = triangle_geometry(centroids_2[label2], centroids_2[k_neighbors_labels_2[n_i]], centroids_2[k_neighbors_labels_2[n_j]])
            cost = cost_geometry_constraints(sides_1, angles_1, sides_2, angles_2)
            costs_geo_neighbors.append(cost)
    
        costs_geo_neighbors = np.mean(costs_geo_neighbors)
        if costs_geo_neighbors < max_cost_geo_neighbors:
            matched_labels_local.append((label1, label2))

            # update the references
            prediction_update_ref[max_id1,:] = 0
            prediction_update_ref[:,max_id2] = 0
            matrix_cost_ref = matrix_cost_ref * prediction_update_ref
    
            # reinitiate matrix_cost_pseudo
            matrix_cost_pseudo = matrix_cost_ref.copy()

        else:
            matrix_patience_ref[max_id1, max_id2] += 1
            if matrix_patience_ref[max_id1, max_id2] > patience_label:
                prediction_update_ref[max_id1, max_id2] = 0
                matrix_cost_ref = matrix_cost_ref * prediction_update_ref
                matrix_cost_pseudo = matrix_cost_pseudo * prediction_update_ref
            matrix_cost_pseudo[max_id1, max_id2] = 0
    
        if np.max(matrix_cost_pseudo)==0:
            matrix_cost_pseudo = matrix_cost_ref.copy()
    
    return matched_labels_local
    
def update_final_prediction(matched_pairs_pseudo, scores, spatial_dist, min_cls_logit, label2index):
        
    # Initiate the prediction
    matched_labels = []
    matched_indexes = []
    matched_labels_dict = {}
    prediction = np.zeros(scores.shape) # 1: confirmed pair
    prediction_update = np.ones(scores.shape) # 0: ignore this pair
    
    # Update the prediction
    for annotated_label_pair in tqdm(matched_pairs_pseudo):
        label1, label2 = annotated_label_pair
        max_id1, max_id2 = label2index['img1'][label1], label2index['img2'][label2]
        
        matched_labels.append((label1, label2))
        matched_indexes.append((max_id1, max_id2))
        matched_labels_dict[label1] = label2
        prediction[max_id1, max_id2] = 1
        prediction_update[max_id1,:] = 0
        prediction_update[:,max_id2] = 0
        
    matrix_cost = (scores + spatial_dist) * (scores > min_cls_logit) * prediction_update
    
    return matched_labels, matched_indexes, matched_labels_dict, prediction, prediction_update, matrix_cost

def filter_matched_pairs(
    matched_pairs_pseudo,
    label2index,
    D1, D2,
    n_neighbors_validation,
    centroids_1, centroids_2,
    max_cost_geo_neighbors
):
    # remove the duplicate (false-positive)
    matched_l1 = [l1 for l1,l2 in matched_pairs_pseudo]
    matched_l2 = [l2 for l1,l2 in matched_pairs_pseudo]
    duplicate1 = [l for l in matched_l1 if matched_l1.count(l)>1]
    duplicate2 = [l for l in matched_l2 if matched_l2.count(l)>1]
    matched_pairs_pseudo = [(l1,l2) for l1,l2 in matched_pairs_pseudo if ((l1 not in duplicate1) and (l2 not in duplicate2))]
    
    # filter the false-positive matched pair
    match_labels_1 = [l1 for l1,l2 in matched_pairs_pseudo]
    match_labels_2 = [l2 for l1,l2 in matched_pairs_pseudo]
    matched_labels_dict = {l1:l2 for l1,l2 in matched_pairs_pseudo} 
    matched_indexes_1 = [label2index["img1"][l1] for l1,l2 in matched_pairs_pseudo]
    
    match_labels_filtered = []
    
    for label1 in tqdm(match_labels_1):
        label2 = matched_labels_dict[label1]
        idx1, idx2 = label2index["img1"][label1], label2index["img2"][label2]
        
        # distance of all matched labels in img1 to label1
        dist_1 = D1[np.ix_([idx1], matched_indexes_1)].flatten()
        neighbors_idx_1 = np.argsort(dist_1)[1:1+n_neighbors_validation*2] # remove self
        neighbors_idx_1 = list(neighbors_idx_1)
        neighbors_idx_1 = random.sample(neighbors_idx_1, k=n_neighbors_validation) 
        neighbors_labels_1 = [match_labels_1[i] for i in neighbors_idx_1]
        
        # match the ROI in img2
        neighbors_labels_2 = [matched_labels_dict[l] for l in neighbors_labels_1]
        neighbors_idx_2 = [label2index["img2"][l2] for l2 in neighbors_labels_2]
        
        # cost geometry of neighbors
        costs_geo_neighbors = []
        for n_i, n_j in list(itertools.combinations(range(n_neighbors_validation), 2)):
            sides_1, angles_1 = triangle_geometry(centroids_1[label1], centroids_1[neighbors_labels_1[n_i]], centroids_1[neighbors_labels_1[n_j]])
            sides_2, angles_2 = triangle_geometry(centroids_2[label2], centroids_2[neighbors_labels_2[n_i]], centroids_2[neighbors_labels_2[n_j]])
            cost = cost_geometry_constraints(sides_1, angles_1, sides_2, angles_2)
            costs_geo_neighbors.append(cost)
        
        costs_geo_neighbors = np.mean(costs_geo_neighbors)
        if costs_geo_neighbors < max_cost_geo_neighbors:
            match_labels_filtered.append((label1,label2))

    return match_labels_filtered

def geometry_constraints(triple_indexes, pseudo_centroid_1, pseudo_centroid_2):
    i1,i2,i3 = triple_indexes
    sides_1, angles_1 = triangle_geometry(pseudo_centroid_1[i1], pseudo_centroid_1[i2], pseudo_centroid_1[i3])
    sides_2, angles_2 = triangle_geometry(pseudo_centroid_2[i1], pseudo_centroid_2[i2], pseudo_centroid_2[i3])
    cost = cost_geometry_constraints(sides_1, angles_1, sides_2, angles_2)
    return cost

def cost_geo_kROIs(args):
    k_indexes, pseudo_centroid_1, pseudo_centroid_2 = args
    return [
        geometry_constraints(triplet, pseudo_centroid_1, pseudo_centroid_2)
        for triplet in list(itertools.combinations(k_indexes,3))
    ]

def compute_geo_costs_parallel(n_initial_guess, n_pair_selected, n_processes, pseudo_centroid_1, pseudo_centroid_2):
    kROIs_combinations = list(itertools.combinations(range(n_initial_guess), n_pair_selected))
    args_list = [(k_indexes, pseudo_centroid_1, pseudo_centroid_2) for k_indexes in kROIs_combinations]
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        costs = pool.map(cost_geo_kROIs, args_list)
    return kROIs_combinations, costs

def filter_pair(label1, label2, matched_labels, D1, D2, centroids_1, centroids_2, label2index, n_neighbors_validation, max_cost_geo_neighbors):
    idx1, idx2 = label2index["img1"][label1], label2index["img2"][label2]

    match_labels_1 = [l1 for l1,l2 in matched_labels]
    match_labels_2 = [l2 for l1,l2 in matched_labels]
    matched_labels_dict = {l1:l2 for l1,l2 in matched_labels} 
    matched_indexes_1 = [label2index["img1"][l1] for l1,l2 in matched_labels]
    
    # distance of all matched labels in img1 to label1
    dist_1 = D1[np.ix_([idx1], matched_indexes_1)].flatten()
    neighbors_idx_1 = np.argsort(dist_1)[1:1+n_neighbors_validation*2] # remove self
    neighbors_idx_1 = list(neighbors_idx_1)
    neighbors_idx_1 = random.sample(neighbors_idx_1, k=n_neighbors_validation) 
    neighbors_labels_1 = [match_labels_1[i] for i in neighbors_idx_1]
    
    # match the ROI in img2
    neighbors_labels_2 = [matched_labels_dict[l] for l in neighbors_labels_1]
    neighbors_idx_2 = [label2index["img2"][l2] for l2 in neighbors_labels_2]
    
    # cost geometry of neighbors
    costs_geo_neighbors = []
    for n_i, n_j in list(itertools.combinations(range(n_neighbors_validation), 2)):
        sides_1, angles_1 = triangle_geometry(centroids_1[label1], centroids_1[neighbors_labels_1[n_i]], centroids_1[neighbors_labels_1[n_j]])
        sides_2, angles_2 = triangle_geometry(centroids_2[label2], centroids_2[neighbors_labels_2[n_i]], centroids_2[neighbors_labels_2[n_j]])
        cost = cost_geometry_constraints(sides_1, angles_1, sides_2, angles_2)
        costs_geo_neighbors.append(cost)
    
    costs_geo_neighbors = np.mean(costs_geo_neighbors)
    return costs_geo_neighbors < max_cost_geo_neighbors

def match_fibers(
    img1, list_label_1,
    img2, list_label_2,
    label2index,
    dir_embedding,
    images_dir, 
    cls_checkpoint,
    device,
    list_k = [3,5,7],

    # initial guess
    n_initial_guess = 80,
    n_pair_selected = 4,
    max_cost_geo_initial = 0.1,
    min_cls_logit_init = 0.6,

    # local prediction
    n_neighbors_ref_init = 256,
    n_neighbors_ref_min = 32,
    max_cost_geo_neighbors = 1,
    min_cls_logit = 0.25,
    patience_label = 5,    
    n_neighbors_validation =3,
    n_processes = 60,
):
    # load classifier
    classifier = PairClassifier().to(device)
    classifier.load_state_dict(torch.load(cls_checkpoint))
    classifier.eval()
    
    # Get different cost matrixes
    print("   Get classification logits and local spatial cost matrix")
    scores, spatial_dist, neighbors_dicts, cp_outputs = GetCostMatrix(
        img1, list_label_1,
        img2, list_label_2,
        classifier,
        dir_embedding,
        images_dir, 
        list_k,
    )

    neighbors_dict_1, neighbors_dict_2 = neighbors_dicts
    cp_output_1, cp_output_2 = cp_outputs
    
    # dictionaries of centroid coordinates
    centroids_1 = get_centroid_dict(cp_output_1[1], list_label_1)
    centroids_2 = get_centroid_dict(cp_output_2[1], list_label_2)
    
    # distance matrix
    D1 = cdist(list(centroids_1.values()), list(centroids_1.values()), metric='euclidean')
    D2 = cdist(list(centroids_2.values()), list(centroids_2.values()), metric='euclidean')

    # make n_initial_guess prediction of the pairs with highest scores
    selected_combs = []
    n_try_init = 0
    # require at least 3 annotated pairs for the next step
    while (len(selected_combs)<3) and (n_try_init<3): 
        n_initial_guess = min([n_initial_guess, len(list_label_1), len(list_label_2)])
        
        # initiate cost matrix
        matrix_cost_pseudo = (scores + spatial_dist) * (scores > min_cls_logit_init)
        
        # initiate attention mask and prediction
        prediction_update_pseudo = np.ones(scores.shape)
        prediction_pseudo = np.zeros(scores.shape)
        
        matched_pairs_pseudo = []
        
        print(f"   Making {n_initial_guess} initial prediction with highest scores")
        while len(matched_pairs_pseudo) < n_initial_guess:
            max_id1, max_id2 = np.unravel_index(np.argmax(matrix_cost_pseudo), matrix_cost_pseudo.shape)
            label1, label2 = list_label_1[max_id1], list_label_2[max_id2]
            
            prediction_update_pseudo[max_id1,:] = 0
            prediction_update_pseudo[:,max_id2] = 0
            prediction_pseudo[max_id1, max_id2] = 1
            matrix_cost_pseudo = matrix_cost_pseudo * prediction_update_pseudo
            matched_pairs_pseudo.append((label1, label2))
        
        pseudo_centroid_1 = [centroids_1[label1] for label1, label2 in matched_pairs_pseudo]
        pseudo_centroid_2 = [centroids_2[label2] for label1, label2 in matched_pairs_pseudo]
    
        print(f"   Select and filter groups of {n_pair_selected} pairs from {n_initial_guess} initial guess")       
        kROIs_combinations, costs = compute_geo_costs_parallel(n_initial_guess, n_pair_selected, n_processes, pseudo_centroid_1, pseudo_centroid_2)
        costs = np.array(costs)
        # all values are less than max_cost_geo_initial
        mask_mean = costs.mean(axis=1) < max_cost_geo_initial
        mask_all = np.all(costs < max_cost_geo_initial, axis=1)
        selected_indices = np.where(mask_mean & mask_all)[0]
        # print(f"      Best selected kROIs with min={costs.mean(axis=1)[selected_indices[0]]:.4f}, max={costs.mean(axis=1)[selected_indices[-1]]:.4f}")
        selected_combs = [kROIs_combinations[i] for i in selected_indices]
        selected_combs = list(set([i for comb in selected_combs for i in comb]))
    
        if len(selected_combs) < 3:
            n_initial_guess += 20
            n_try_init += 1
        else:
            print(f"   Select {len(selected_combs)} pairs passed all filters")

    if (n_try_init>=3) and (len(selected_combs)<3):
        print("VERIFY THE IMAGE PAIRS!!!")
    else:
        matched_pairs_pseudo = [matched_pairs_pseudo[i] for i in selected_combs]
    
        # initiate the prediction local
        matched_labels, matched_indexes, matched_labels_dict, prediction, prediction_update, matrix_cost = \
            update_final_prediction(matched_pairs_pseudo, scores, spatial_dist, min_cls_logit, label2index)
    
        # local prediction
        print("   Start local prediction")        
        n_current_prediction = len(matched_pairs_pseudo)
        count_pairs_unchanged = 0
        final_prediction = []
        n_local_prediction = np.inf
    
        # track_local_prediction = {l1:0 for l1 in list_label_1}
        n_step_match = 1
        while (n_local_prediction > 0):
            n_neighbors_ref = int(n_neighbors_ref_init / 2**(n_step_match-1))
            n_neighbors_ref = min([n_neighbors_ref, len(list_label_1), len(list_label_2)])
            n_neighbors_ref = max(n_neighbors_ref, n_neighbors_ref_min)
            print(f'   Local prediction for {n_current_prediction} annotated pairs with {n_neighbors_ref} neighbors')
            args_list = [
                (
                    matched_label,
                    list_label_1, list_label_2,
                    D1, D2,
                    n_neighbors_ref,
                    prediction_update, matrix_cost,
                    matched_indexes, matched_labels_dict,
                    centroids_1, centroids_2,
                    max_cost_geo_neighbors,
                    patience_label,
                )
                for matched_label in matched_pairs_pseudo
            ]
            with multiprocessing.Pool(processes=n_processes) as pool:
                matched_labels_global = pool.map(local_prediction_task, args_list)
            
            matched_pairs_pseudo = list(set(matched_pairs_pseudo + [i for p in matched_labels_global for i in p]))
        
            matched_pairs_pseudo = filter_matched_pairs(
                matched_pairs_pseudo,
                label2index,
                D1, D2,
                n_neighbors_validation,
                centroids_1, centroids_2,
                max_cost_geo_neighbors,
            )
            
            # Reinitiate the prediction
            matched_labels, matched_indexes, matched_labels_dict, prediction, prediction_update, matrix_cost = \
                update_final_prediction(matched_pairs_pseudo, scores, spatial_dist, min_cls_logit, label2index)
        
            # update the tracking values
            n_local_prediction = len(matched_labels) - n_current_prediction
            n_current_prediction = len(matched_labels)
            if sorted(matched_labels) == sorted(final_prediction):
                count_pairs_unchanged += 1
                
            final_prediction = matched_labels
            print(f'      Step {n_step_match}: Found {n_local_prediction} additional pairs, count_pairs_unchanged={count_pairs_unchanged}')
            
            n_step_match += 1

        ## fill the unannotated pairs
        labels1_unannot = [l for l in list_label_1 if l not in [i[0] for i in matched_labels]]
        labels2_unannot = [l for l in list_label_2 if l not in [i[1] for i in matched_labels]]
        
        idx1_unannot = [label2index['img1'][l] for l in labels1_unannot]
        idx2_unannot = [label2index['img2'][l] for l in labels2_unannot]
        print(f"   {len(labels1_unannot)}/{len(labels2_unannot)} ROIs yet unannotated.")

        # exhaustively go through all the remaining possible pairs
        matrix_cost_pseudo = matrix_cost.copy()
        matrix_patience = np.zeros(scores.shape) # number of times the pair is evaluated but didn't pass the filters

        initial_nonzeros = np.count_nonzero(matrix_cost)
        progress_bar = tqdm(total=initial_nonzeros, desc="Remaining non-zero entries", unit="pairs")
        previous_nonzeros = initial_nonzeros
        
        while np.max(matrix_cost) > 0:
            max_id1, max_id2 = np.unravel_index(np.argmax(matrix_cost_pseudo), matrix_cost_pseudo.shape)
            label1, label2 = list_label_1[max_id1], list_label_2[max_id2]
            
            pair_validation = filter_pair(
                label1, label2,
                matched_labels, 
                D1, D2, 
                centroids_1, centroids_2, 
                label2index, 
                n_neighbors_validation,
                max_cost_geo_neighbors
            )
            if pair_validation:
                matched_labels.append((label1, label2))
                prediction_update[max_id1,:] = 0
                prediction_update[:,max_id2] = 0
                matrix_cost = matrix_cost * prediction_update
        
                # reinitiate pseudo matrix
                matrix_cost_pseudo = matrix_cost.copy()
            else:
                matrix_cost_pseudo[max_id1, max_id2] = 0
        
                # follow the times the pairs being evaluated
                matrix_patience[max_id1, max_id2] += 1
                if matrix_patience[max_id1, max_id2] > patience_label:
                    # ignore the pair from final prediction
                    prediction_update[max_id1, max_id2] = 0
                    matrix_cost = matrix_cost * prediction_update
        
                    # update the pseudo matrix to continue
                    matrix_cost_pseudo = matrix_cost_pseudo * prediction_update
                    # print(f"Ignore pair {label1}-{label2}, cost_cls={scores[max_id1, max_id2]:.4f}, cost_dist={spatial_dist[max_id1, max_id2]:.4f}")
           
            if np.max(matrix_cost_pseudo)==0:
                matrix_cost_pseudo = matrix_cost.copy()

            # update the progress bar
            current_nonzeros = np.count_nonzero(matrix_cost)
            progress_bar.update(previous_nonzeros - current_nonzeros)
            previous_nonzeros = current_nonzeros

        progress_bar.close()

        matched_labels_f = filter_matched_pairs(
            matched_labels,
            label2index,
            D1, D2,
            n_neighbors_validation,
            centroids_1, centroids_2,
            max_cost_geo_neighbors, 
        )
        print(f"   Prediction end: {len(matched_labels_f)} pairs found!")
    
        return matched_labels_f, scores, spatial_dist, cp_outputs


def plot_FM_prediction(
    path_img1,
    path_img2,
    matched_labels,
    dpi=100,
):
    def label_match(label_id, matched_labels, img_idx=0):
        l1,l2 = [p for p in matched_labels if p[img_idx]==label_id][0]
        return f"{l1}-{l2}"
        
    cp_output1 = segment_image(path_img1)
    mask1, p1, _ = cp_output1
    cp_output2 = segment_image(path_img2)
    mask2, p2, _ = cp_output2
    
    labels1_filtered = filter_ROIs(cp_output1, size_crop)
    labels2_filtered = filter_ROIs(cp_output2, size_crop)

    image1 = io.imread(path_img1)
    image2 = io.imread(path_img2)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    list_roi1 = [l1 for l1,l2 in matched_labels]
    list_roi2 = [l2 for l1,l2 in matched_labels]

    # Proportional widths
    total_width = w1 + w2
    fig_width = total_width / dpi
    fig_height = max(h1, h2) / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = GridSpec(1, 2, width_ratios=[w1, w2])

    for i, (img, mask, ax_idx) in enumerate(zip([image1, image2], [mask1, mask2], [0, 1])):
        ax = fig.add_subplot(gs[ax_idx])
        ax.imshow(img if img.ndim == 3 else img, cmap='gray')
        
        for region in regionprops(mask):
            # plot the contours in red
            contours = find_contours(mask == region.label, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

            # plot the label
            y, x = region.centroid

            # labels found in matched pairs
            if (ax_idx==0):
                if (region.label in list_roi1):
                    ax.text(x, y, 
                            label_match(region.label, matched_labels, ax_idx), 
                            color='white', fontsize=10,
                            ha='center', va='center',fontweight='bold'
                           )
                elif (region.label in labels1_filtered):
                    ax.text(x, y, 
                            str(region.label), 
                            color='red', fontsize=10,
                            ha='center', va='center',
                           )
                else:
                    ax.text(x, y, str(region.label), color='gray', fontsize=8,
                            ha='center', va='center')
                    
            if (ax_idx==1):
                if (region.label in list_roi2):
                    ax.text(x, y, 
                            label_match(region.label, matched_labels, ax_idx), 
                            color='white', fontsize=10,
                            ha='center', va='center', fontweight='bold'
                           )
                elif (region.label in labels2_filtered):
                    ax.text(x, y, 
                            str(region.label), 
                            color='red', fontsize=10,
                            ha='center', va='center',
                           )
                else:
                    ax.text(x, y, str(region.label), color='gray', fontsize=8,
                            ha='center', va='center')

        ax.set_aspect('equal')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

    return [cp_output1, labels1_filtered], [cp_output2, labels2_filtered]

# def save_FM_prediction(
#     img,
#     images_dir,
#     img_index, #0/1: 1st/2nd image in the pair
#     output_plot_prediction, # output from plot_FM_prediction()
#     matched_labels,
#     dir_save_prediction_output,
#     dpi=80,
# ):
#     (masks, _, _), labels_filtered = output_plot_prediction
#     list_roi_matched = [l[img_index] for l in matched_labels]

#     # plot image
#     image = io.imread(f"{images_dir}/{img}.png")
#     h, w = image.shape[:2]

#     # Create figure at correct size
#     figsize = (w / dpi, h / dpi)
#     fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
#     ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
#     ax.axis('off')

#     def label_match(label_id, matched_labels, img_idx=0):
#         l1,l2 = [p for p in matched_labels if p[img_idx]==label_id][0]
#         return f"{l1}-{l2}"

#     for region in tqdm(regionprops(masks)):
#         # plot the contours in red
#         contours = find_contours(masks == region.label, 0.5)
#         for contour in contours:
#             ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

#         # plot the label
#         y, x = region.centroid

#         # labels found in matched pairs
#         if (region.label in list_roi_matched):
#             ax.text(x, y, 
#                     label_match(region.label, matched_labels, img_index), 
#                     color='white', fontsize=10,
#                     ha='center', va='center',fontweight='bold'
#                    )
#         elif (region.label in labels_filtered):
#             ax.text(x, y, 
#                     str(region.label), 
#                     color='red', fontsize=10,
#                     ha='center', va='center',
#                    )
#         else:
#             ax.text(x, y, str(region.label), color='gray', fontsize=8,
#                     ha='center', va='center')
    
#     # Save with no padding or extra margins
#     fig.savefig(f"{dir_save_prediction_output}/{img}_MF_img{img_index}.png", 
#                 dpi=dpi, 
#                 bbox_inches='tight', 
#                 pad_inches=0
#                )
#     plt.close(fig)

def prepare_region_annotation(args):
    region_label, masks, matched_labels, labels_filtered, img_index = args
    contours = find_contours(masks == region_label, 0.5)
    region_mask = masks == region_label
    y, x = np.argwhere(region_mask).mean(axis=0)

    def label_match(label_id):
        l1, l2 = [p for p in matched_labels if p[img_index] == label_id][0]
        return f"{l1}-{l2}"

    if region_label in [p[img_index] for p in matched_labels]:
        text = label_match(region_label)
        color = 'white'
        weight = 'bold'
        fontsize = 10
    elif region_label in labels_filtered:
        text = str(region_label)
        color = 'red'
        weight = 'normal'
        fontsize = 10
    else:
        text = str(region_label)
        color = 'gray'
        weight = 'normal'
        fontsize = 8

    return {
        'contours': contours,
        'text': text,
        'coords': (x, y),
        'color': color,
        'weight': weight,
        'fontsize': fontsize,
    }

def save_FM_prediction(
    img,
    images_dir,
    img_index, #0/1: 1st/2nd image in the pair
    output_plot_prediction, # output from plot_FM_prediction()
    matched_labels,
    dir_save_prediction_output,
    dpi=80,
):
    (masks, _, _), labels_filtered = output_plot_prediction
    list_roi_matched = [l[img_index] for l in matched_labels]
    
    region_labels = [r.label for r in regionprops(masks)]
    args_list = [
        (label, masks, matched_labels, labels_filtered, img_index)
        for label in region_labels
    ]
    
    # with multiprocessing.Pool(processes=n_processes) as pool:
    #     draw_instructions = list(pool.map(prepare_region_annotation, args_list))
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        draw_instructions = list(
            tqdm(
                pool.imap_unordered(prepare_region_annotation, args_list),
                total=len(args_list),
                desc="Processing regions"
            )
        )
    
    # plot image
    image = io.imread(f"{images_dir}/{img}.png")
    h, w = image.shape[:2]
    
    # Create figure at correct size
    figsize = (w / dpi, h / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    ax.axis('off')
    
    for item in tqdm(draw_instructions):
        for contour in item['contours']:
            ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)
        x, y = item['coords']
        ax.text(x, y, 
                item['text'], 
                color=item['color'],
                fontsize=item['fontsize'],
                ha='center', 
                va='center',
                fontweight=item['weight']
               )
    
    fig.savefig(
        f"{dir_save_prediction_output}/{img}_MF_img{img_index}.png", 
        dpi=dpi,
        pad_inches=0,
    )
    plt.close(fig)

## Prepare datasets
# 1. Generate npz files as VAE inputs
def FM_generate_VAE_inputs(
    list_images,
    images_dir,
    dir_save_npz
):
    for img_name in list_images:
        image_path = f"{images_dir}/{img_name}.png"
        
        # segmentation by CP
        cp_output = segment_image(image_path)
        masks, props, img_rec = cp_output
        labels = filter_ROIs(cp_output, size_crop)

        print(f"Generate {len(labels)} npz files")
        for label_id in tqdm(labels):
            # original
            x1, x2 = crop_stack_mask_original(cp_output, label_id)
            np.savez_compressed(
                f"{dir_save_npz}/{img_name}_roi_{label_id}.npz",
                flow_x=x1[...,0],
                flow_y=x1[...,1],
                cell_prob=x1[...,2],
                roi_mask=x2[...,0]
            )

# 2. Generate and save embedding 
def FM_generate_embedding(
    dir_save_npz,
    dir_embedding,
    VAE_checkpoint,
    device
):
    vae = SharedMultiHeadVAE().to(device)
    vae.load_state_dict(torch.load(VAE_checkpoint))
    vae.eval()
    
    list_path_npz_files = [f"{dir_save_npz}/{f}" for f in os.listdir(dir_save_npz)]
    ds = DatasetForEmbeddingExtraction(list_path_npz_files)
    
    def save_embeddings(dataset, vae, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        vae.eval()
        device = next(vae.parameters()).device
    
        for i in tqdm(range(len(dataset))):
            path = dataset.files[i]
            base = os.path.basename(path).replace(".npz", ".npy")
            save_path = os.path.join(save_dir, base)
            if not os.path.exists(save_path):
                tensor = dataset.create_input(path).unsqueeze(0).to(device)
                with torch.no_grad():
                    mu = vae.encode(tensor).squeeze(0).cpu().numpy()
                np.save(save_path, mu)
    
    save_embeddings(ds, vae, save_dir=dir_embedding)