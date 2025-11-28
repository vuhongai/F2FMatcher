from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm
import math
import itertools
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
from skimage.transform import AffineTransform

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
    img1, list_label_1, cp_output_1,
    img2, list_label_2, cp_output_2,
    classifier,
    dir_embedding,
    images_dir, 
    dir_save_cellpose_masks, 
    list_k = [3,5,7],
    CP_model_name_1=None,
    CP_model_name_2=None,
):
    # get classification logits
    scores = GetClassifierLogitsFromLatentVector(
        img1, list_label_1,
        img2, list_label_2,
        classifier,
        dir_embedding,
    )

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
    cost_sides = mean_absolute_error(sides_1, sides_2)
    cost_angles = mean_absolute_error(angles_1/180, angles_2/180)
    return cost_sides * cost_angles

def costs_geometry(sides_1, angles_1, sides_2, angles_2):
    cost_sides = mean_absolute_error(sides_1, sides_2)
    cost_angles = mean_absolute_error(angles_1/180, angles_2/180)
    return cost_sides, cost_angles

def get_k_nearest_neighbor(D, idx, k=3):
    distances = D[idx]
    distances[idx] = np.inf
    return np.argsort(distances)[:k]

def get_neighbors_ref(Di, idi_ref, list_label_i, n_neighbors_ref):
    dist_ref_i = Di[idi_ref,:]
    neighbors_ref_id_i = [i for i in np.argsort(dist_ref_i)[:n_neighbors_ref]]
    neighbors_ref_label_i = [list_label_i[i] for i in neighbors_ref_id_i]
    return neighbors_ref_id_i, neighbors_ref_label_i

def get_neighbors_ref_by_distance(Di, idi_ref, list_label_i, distance_neighbors_ref):
    dist_ref_i = Di[idi_ref,:]
    neighbors_ref_id_i = np.where(dist_ref_i < distance_neighbors_ref)[0]
    neighbors_ref_label_i = [list_label_i[i] for i in neighbors_ref_id_i]
    return neighbors_ref_id_i, neighbors_ref_label_i

def estimate_affine_transform(P_list, Q_list):
    """
    Estimate affine transform from points P to Q
    Inputs:
        P_list: list of (y, x) in image1 (result of region.centroid)
        Q_list: list of (y, x) in image2
    Returns:
        transform: skimage AffineTransform object
    """
    P_array = np.array(P_list)
    Q_array = np.array(Q_list)
    tform = AffineTransform()
    tform.estimate(P_array, Q_array)
    return tform

def local_prediction_task(args):
    matched_label, \
    list_label_1, list_label_2, \
    label2index, \
    D1, D2, \
    distance_neighbors_ref, \
    prediction_update, matrix_cost, \
    matched_indexes, matched_labels_dict, centroids_1, centroids_2, \
    max_cost_geo_neighbors_sides, max_cost_geo_neighbors_angles, \
    patience_label = args

    label1_ref, label2_ref = matched_label
    id1_ref, id2_ref = label2index['img1'][label1_ref], label2index['img2'][label2_ref]

    neighbors_ref_id_1, neighbors_ref_label_1 = get_neighbors_ref_by_distance(D1, id1_ref, list_label_1, distance_neighbors_ref)
    neighbors_ref_id_2, neighbors_ref_label_2 = get_neighbors_ref_by_distance(D2, id2_ref, list_label_2, distance_neighbors_ref)

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
    
        costs_neighbors_sides, costs_neighbors_angles = [], []
        for n_i, n_j in [(0,1), (0,2), (1,2)]:
            sides_1, angles_1 = triangle_geometry(centroids_1[label1], centroids_1[k_neighbors_labels_1[n_i]], centroids_1[k_neighbors_labels_1[n_j]])
            sides_2, angles_2 = triangle_geometry(centroids_2[label2], centroids_2[k_neighbors_labels_2[n_i]], centroids_2[k_neighbors_labels_2[n_j]])
            cost_sides, cost_angles = costs_geometry(sides_1, angles_1, sides_2, angles_2)
            costs_neighbors_sides.append(cost_sides)
            costs_neighbors_angles.append(cost_angles)

        if (np.mean(costs_neighbors_sides) < max_cost_geo_neighbors_sides) and (np.mean(costs_neighbors_angles) < max_cost_geo_neighbors_angles):
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
    max_cost_geo_neighbors_sides,
    max_cost_geo_neighbors_angles,
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
        neighbors_idx_1 = np.argsort(dist_1)[1:1+n_neighbors_validation] # remove self
        neighbors_labels_1 = [match_labels_1[i] for i in neighbors_idx_1]
        
        # match the ROI in img2
        neighbors_labels_2 = [matched_labels_dict[l] for l in neighbors_labels_1]
        neighbors_idx_2 = [label2index["img2"][l2] for l2 in neighbors_labels_2]
        
        # cost geometry of neighbors
        costs_neighbors_sides, costs_neighbors_angles = [], []
        for n_i, n_j in list(itertools.combinations(range(n_neighbors_validation), 2)):
            sides_1, angles_1 = triangle_geometry(centroids_1[label1], centroids_1[neighbors_labels_1[n_i]], centroids_1[neighbors_labels_1[n_j]])
            sides_2, angles_2 = triangle_geometry(centroids_2[label2], centroids_2[neighbors_labels_2[n_i]], centroids_2[neighbors_labels_2[n_j]])
            cost_sides, cost_angles = costs_geometry(sides_1, angles_1, sides_2, angles_2)
            costs_neighbors_sides.append(cost_sides)
            costs_neighbors_angles.append(cost_angles)

        if (np.mean(costs_neighbors_sides)<max_cost_geo_neighbors_sides) and (np.mean(costs_neighbors_angles)<max_cost_geo_neighbors_angles):
            match_labels_filtered.append((label1,label2))

    return match_labels_filtered

def geometry_constraints(
    triple_indexes, 
    pseudo_centroid_1, pseudo_centroid_2,
    ):
    i1,i2,i3 = triple_indexes
    sides_1, angles_1 = triangle_geometry(pseudo_centroid_1[i1], pseudo_centroid_1[i2], pseudo_centroid_1[i3])
    sides_2, angles_2 = triangle_geometry(pseudo_centroid_2[i1], pseudo_centroid_2[i2], pseudo_centroid_2[i3])
    cost_sides, cost_angles = costs_geometry(sides_1, angles_1, sides_2, angles_2)
    # return (cost_sides < max_cost_geo_neighbors_sides) * (cost_angles < max_cost_geo_neighbors_angles)
    return cost_sides, cost_angles

def cost_geo_kROIs(args):
    k_indexes, \
    pseudo_centroid_1, pseudo_centroid_2 = args
    
    return [
        geometry_constraints(
            triplet, 
            pseudo_centroid_1, pseudo_centroid_2
            )
        for triplet in list(itertools.combinations(k_indexes,3))
    ]

def compute_geo_costs_parallel(
    n_initial_guess, 
    n_pair_selected, 
    n_processes, 
    pseudo_centroid_1, pseudo_centroid_2,
    ):
    kROIs_combinations = list(itertools.combinations(range(n_initial_guess), n_pair_selected))
    args_list = [
        (k_indexes, pseudo_centroid_1, pseudo_centroid_2) \
            for k_indexes in kROIs_combinations]
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        costs = pool.map(cost_geo_kROIs, args_list)
    return kROIs_combinations, costs

def remove_duplicate_matched_labels(matched_pairs_pseudo):
    # remove the duplicate (false-positive)
    matched_l1 = [l1 for l1,l2 in matched_pairs_pseudo]
    matched_l2 = [l2 for l1,l2 in matched_pairs_pseudo]
    duplicate1 = [l for l in matched_l1 if matched_l1.count(l)>1]
    duplicate2 = [l for l in matched_l2 if matched_l2.count(l)>1]
    matched_pairs_pseudo = [(l1,l2) for l1,l2 in matched_pairs_pseudo if ((l1 not in duplicate1) and (l2 not in duplicate2))]
    return matched_pairs_pseudo

def filter_pair(
    label1, label2, 
    matched_labels, 
    D1, D2, 
    centroids_1, centroids_2, 
    label2index, 
    n_neighbors_validation, 
    max_cost_geo_neighbors_sides,
    max_cost_geo_neighbors_angles,
):
    idx1, idx2 = label2index["img1"][label1], label2index["img2"][label2]

    match_labels_1 = [l1 for l1,l2 in matched_labels]
    match_labels_2 = [l2 for l1,l2 in matched_labels]
    matched_labels_dict = {l1:l2 for l1,l2 in matched_labels} 
    matched_indexes_1 = [label2index["img1"][l1] for l1,l2 in matched_labels]
    
    # distance of all matched labels in img1 to label1
    dist_1 = D1[np.ix_([idx1], matched_indexes_1)].flatten()
    neighbors_idx_1 = np.argsort(dist_1)[1:1+n_neighbors_validation] # remove self
    neighbors_labels_1 = [match_labels_1[i] for i in neighbors_idx_1]
    
    # match the ROI in img2
    neighbors_labels_2 = [matched_labels_dict[l] for l in neighbors_labels_1]
    neighbors_idx_2 = [label2index["img2"][l2] for l2 in neighbors_labels_2]
    
    # cost geometry of neighbors
    costs_neighbors_sides, costs_neighbors_angles = [], []
    for n_i, n_j in list(itertools.combinations(range(n_neighbors_validation), 2)):
        sides_1, angles_1 = triangle_geometry(centroids_1[label1], centroids_1[neighbors_labels_1[n_i]], centroids_1[neighbors_labels_1[n_j]])
        sides_2, angles_2 = triangle_geometry(centroids_2[label2], centroids_2[neighbors_labels_2[n_i]], centroids_2[neighbors_labels_2[n_j]])
        cost_sides, cost_angles = costs_geometry(sides_1, angles_1, sides_2, angles_2)
        costs_neighbors_sides.append(cost_sides)
        costs_neighbors_angles.append(cost_angles)
    
    if (np.mean(costs_neighbors_sides)<max_cost_geo_neighbors_sides) and (np.mean(costs_neighbors_angles)<max_cost_geo_neighbors_angles):
        return True
    else:
        return False

def match_fibers(
    img1, list_label_1, cp_output_1,
    img2, list_label_2, cp_output_2,
    label2index,
    dir_embedding,
    images_dir, 
    dir_save_cellpose_masks,
    cls_checkpoint,
    device,
    list_k = [3,5,7],

    # initial guess
    n_initial_guess = 80,
    n_pair_selected = 4,
    min_cls_logit_init = 0.75,

    # local prediction
    distance_neighbors_ref = 200,
    max_distance_affine = 200,
    max_cost_geo_neighbors_sides = 30,
    max_cost_geo_neighbors_angles = 0.15,
    min_cls_logit = 0.5,
    patience_label = 5,    
    n_neighbors_validation = 3,
    n_processes = 60,
    n_try_unannotated = 1,

    # Cellpose model
    CP_model_name_1=None,
    CP_model_name_2=None,

    # Save predicted pairs in each step for visualization
    save_step_prediction=False,
    dir_save_prediction_output=None,
):
    # Initiate the prediction
    if save_step_prediction:
        step_prediction = {}

    # load classifier
    classifier = PairClassifier().to(device)
    classifier.load_state_dict(torch.load(cls_checkpoint))
    classifier.eval()
    
    # Get different cost matrixes
    print("   Get classification logits and local spatial cost matrix")
    scores, spatial_dist, neighbors_dicts, _ = GetCostMatrix(
        img1, list_label_1, cp_output_1,
        img2, list_label_2, cp_output_2,
        classifier,
        dir_embedding,
        images_dir, 
        dir_save_cellpose_masks,
        list_k,
        CP_model_name_1,
        CP_model_name_2
    )

    neighbors_dict_1, neighbors_dict_2 = neighbors_dicts
    
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
        if save_step_prediction:
            step_prediction["1_initial_guess"] = matched_pairs_pseudo

        kROIs_combinations, costs = compute_geo_costs_parallel(n_initial_guess, n_pair_selected, n_processes, pseudo_centroid_1, pseudo_centroid_2)
        costs = np.array(costs)

        # select the combinations of n_pair_selected that pass all the filters of sides and angles
        costs_sides = (costs[...,0]<max_cost_geo_neighbors_sides).sum(axis=-1)
        costs_angles = (costs[...,1]<max_cost_geo_neighbors_angles).sum(axis=-1)

        n_comb = math.comb(n_pair_selected,3)
        selected_indices = np.where((costs_sides==n_comb) & (costs_angles==n_comb))[0]
        # print(f"      Best selected kROIs with min={costs.mean(axis=1)[selected_indices[0]]:.4f}, max={costs.mean(axis=1)[selected_indices[-1]]:.4f}")
        selected_combs = [kROIs_combinations[i] for i in selected_indices]
        selected_combs = list(set([i for comb in selected_combs for i in comb]))
    
        if len(selected_combs) < 3:
            n_initial_guess += 20
            n_try_init += 1
        else:
            print(f"   Found {len(selected_combs)} pairs passed all filters")

    # start local prediction based on the initial seed
    if (n_try_init>=3) and (len(selected_combs)<3):
        print("VERIFY THE IMAGE PAIRS!!!")
    else:
        matched_pairs_pseudo = [matched_pairs_pseudo[i] for i in selected_combs]

        # initiate the prediction local
        matched_labels, matched_indexes, matched_labels_dict, prediction, prediction_update, matrix_cost = \
            update_final_prediction(matched_pairs_pseudo, scores, spatial_dist, min_cls_logit, label2index)

        if save_step_prediction:
            step_prediction["2_selected_combs_from_initial_guess"] = matched_labels

        # local prediction
        print("   Start local prediction")        
        n_current_prediction = len(matched_pairs_pseudo)
        count_pairs_unchanged = 0
        final_prediction = []
        n_local_prediction = np.inf
        n_step_match = 1
        if save_step_prediction:
            step_prediction["3_local_prediction"] = []

        while (n_local_prediction > 0):
            print(f'   Local prediction for {n_current_prediction} annotated pairs')
            args_list = [
                (
                    matched_label, \
                    list_label_1, list_label_2, \
                    label2index, \
                    D1, D2, \
                    distance_neighbors_ref, \
                    prediction_update, matrix_cost, \
                    matched_indexes, matched_labels_dict, centroids_1, centroids_2, \
                    max_cost_geo_neighbors_sides, max_cost_geo_neighbors_angles, \
                    patience_label
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
                max_cost_geo_neighbors_sides,
                max_cost_geo_neighbors_angles,
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
            
            if save_step_prediction:
                step_prediction["3_local_prediction"].append(final_prediction)
            n_step_match += 1

        ## fill the unannotated pairs
        print(f"   Start fill the unannotated ROIs")
        if save_step_prediction:
            step_prediction["4_unannotated_prediction"] = []

        ### check if any unannotated ROIs
        labels1_unannot = [l for l in list_label_1 if l not in [i[0] for i in matched_labels]]
        labels2_unannot = [l for l in list_label_2 if l not in [i[1] for i in matched_labels]]

        if len(labels1_unannot)*len(labels2_unannot) > 0:
            for t in range(n_try_unannotated):
                labels1_unannot = [l for l in list_label_1 if l not in [i[0] for i in matched_labels]]
                labels2_unannot = [l for l in list_label_2 if l not in [i[1] for i in matched_labels]]
                # idx1_unannot = [label2index['img1'][l] for l in labels1_unannot]
                # idx2_unannot = [label2index['img2'][l] for l in labels2_unannot]
                print(f"   {len(labels1_unannot)}/{len(labels2_unannot)} ROIs yet unannotated.")

                # define the affine transformation based on all annotated labels' centroids
                centroids_annotated_1 = [centroids_1[p[0]] for p in matched_labels]
                centroids_annotated_2 = [centroids_2[p[1]] for p in matched_labels]
                tform = estimate_affine_transform(centroids_annotated_1, centroids_annotated_2)

                # calculate the distance of affine-transformed centroids from img1>img2 with all unannotated ROI in img2
                centroids_unannotated_1 = np.array([centroids_1[l] for l in labels1_unannot])
                centroids_unannotated_1pred = tform(centroids_unannotated_1)

                # only cls score is used as matrix cost for unannotated ROIs
                # due to some neighbor is missing in one of the 2 imgs
                scores_unannotated = scores * prediction_update

                # match between 2 lists
                for i,label1 in tqdm(enumerate(labels1_unannot)):
                    _dist_affine_img2 = cdist(
                        [centroids_unannotated_1pred[i]], 
                        [centroids_2[l] for l in labels2_unannot],
                        metric='euclidean'
                    )[0]
                    
                    _costs = scores_unannotated[np.ix_(
                        [label2index['img1'][label1]],
                        [label2index['img2'][l] for l in labels2_unannot]
                    )][0]
                    
                    _indexes_img2 = np.where((_dist_affine_img2 < max_distance_affine) & (_costs > min_cls_logit))[0]
                    if len(_indexes_img2)>0:
                        _costs = _costs[_indexes_img2]
                        # sort the scores
                        _sorted_indices = np.argsort(-_costs)
                        _indexes_img2 = _indexes_img2[_sorted_indices]
                        
                        for _idx2 in _indexes_img2:
                            label2 = labels2_unannot[_idx2]
                            fil = filter_pair(
                                label1, label2, 
                                matched_labels, 
                                D1, D2, 
                                centroids_1, centroids_2, 
                                label2index, 
                                n_neighbors_validation, 
                                max_cost_geo_neighbors_sides,
                                max_cost_geo_neighbors_angles,
                            )
                            if fil:
                                matched_labels.append((label1, label2))
                                labels2_unannot.remove(label2)
                                matched_labels = remove_duplicate_matched_labels(matched_labels)
                                if save_step_prediction:
                                    step_prediction["4_unannotated_prediction"].append(matched_labels)
                                break

                # matched_labels = filter_matched_pairs(
                #     matched_labels,
                #     label2index,
                #     D1, D2,
                #     n_neighbors_validation,
                #     centroids_1, centroids_2,
                #     max_cost_geo_neighbors_sides,
                #     max_cost_geo_neighbors_angles,
                # )
                print(f"      Try {t+1}: Found total {len(matched_labels)} pairs.")
        print(f"   Prediction end: {len(matched_labels)} pairs found! \n")
        if save_step_prediction:
            with open(f"{dir_save_prediction_output}/step_prediction.pkl", "wb") as f:
                pickle.dump(step_prediction, f)
        return matched_labels, scores, spatial_dist, [cp_output_1, cp_output_2]


def prepare_region_annotation(args):
    region_label, masks, matched_labels, labels_filtered, img_index, IHF = args
    contours = find_contours(masks == region_label, 0.5)
    region_mask = masks == region_label
    y, x = np.argwhere(region_mask).mean(axis=0)

    def label_match(label_id):
        l1, l2 = [p for p in matched_labels if p[img_index] == label_id][0]
        return f"{l1}-{l2}"
    
    if region_label in [p[img_index] for p in matched_labels]:
        text = label_match(region_label)
        color = 'white' if IHF else 'black'
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

    color_contour = 'red' if IHF else 'gray'
    return {
        'contours': contours,
        'text': text,
        'coords': (x, y),
        'color': color,
        'weight': weight,
        'fontsize': fontsize,
        'color_contour': color_contour,
    }

def save_FM_prediction(
    img,
    images_dir,
    img_index, #0/1: 1st/2nd image in the pair
    output_plot_prediction, # output from plot_FM_prediction()
    matched_labels,
    dir_save_prediction_output,
    dpi=80,
    IHF=True,
):
    (masks, _, _), labels_filtered = output_plot_prediction
    list_roi_matched = [l[img_index] for l in matched_labels]
    
    region_labels = [r.label for r in regionprops(masks)]
    args_list = [
        (label, masks, matched_labels, labels_filtered, img_index, IHF)
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
            ax.plot(contour[:, 1], contour[:, 0], color=item['color_contour'], linewidth=1)
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
    img_name,
    images_dir,
    dir_save_npz,
    CP_model_name,
    dir_save_cellpose_masks
):
    image_path = f"{images_dir}/{img_name}.png"
    path_mask = f"{dir_save_cellpose_masks}/{CP_model_name}_{img_name}_CP_outputs"
    
    # segmentation by CP
    cp_output = segment_image(
        img_path=image_path, 
        CP_model_name=CP_model_name,
        savedir=dir_save_cellpose_masks
        )
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
    return cp_output

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


# Visualize prediction step-by-step
params_plot_step_prediction = {
    "IHF": {
        "confirmed": {"color_contour":"red", "size_contour":1, "facecolor":"white", "alpha":0.5},
        "on-going": {"color_contour":"red", "size_contour":0.5, "facecolor":"white", "alpha":0.2},
        "ignored": {"color_contour":"black", "size_contour":0.1, "facecolor":"white", "alpha":0.0},
    },
    "BrighField": {
        "confirmed": {"color_contour":"black", "size_contour":1, "facecolor":"red", "alpha":0.5},
        "on-going": {"color_contour":"black", "size_contour":0.5, "facecolor":"red", "alpha":0.2},
        "ignored": {"color_contour":"white", "size_contour":0.1, "facecolor":"red", "alpha":0.0},
    },
}

def prepare_region_annotation_for_step_prediction(args):
    region_label, masks, params_plot = args
    contour = find_contours(masks == region_label, 0.5)
    return {
        'contours': contour,
        'size_contours': params_plot['size_contour'],
        'color_contour': params_plot['color_contour'],
        'facecolor': params_plot['facecolor'],
        'alpha': params_plot['alpha']
    }

def get_list_params_contours(list_rois, masks, IHF, ROI_type):
    image_style = "IHF" if IHF else "BrighField"
    args_list = [
        (label, masks, params_plot_step_prediction[image_style][ROI_type])
        for label in list_rois
    ]
    with multiprocessing.Pool(processes=n_processes) as pool:
        draw_instructions = list(
            tqdm(
                pool.imap_unordered(prepare_region_annotation_for_step_prediction, args_list),
                total=len(args_list),
                desc=f"Processing regions - {ROI_type}"
            )
        )
    return draw_instructions

def save_image_per_step(
    img, images_dir, 
    draw_instructions,
    dir_save_prediction_output,
    suffix,
    dpi=100,
):
    # plot image
    image = io.imread(f"{images_dir}/{img}.png")
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(w/300, h/300), dpi=dpi)
    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    for item in tqdm(draw_instructions):
        for c in item['contours']:
            # c is (y, x); Polygon need (x, y)
            xy = np.c_[c[:, 1], c[:, 0]]
            # filled patch
            ax.add_patch(Polygon(xy, closed=True, facecolor=item['facecolor'], edgecolor='none', alpha=item['alpha']))
            ax.plot(xy[:, 0], xy[:, 1], color=item['color_contour'], linewidth=item['size_contours'])
    
    ax.set_aspect('equal')
    ax.axis('off')
    # plt.show()

    # Save image
    fig.savefig(
        f"{dir_save_prediction_output}/{img}_{suffix}.png", 
        dpi=dpi, bbox_inches='tight', pad_inches=0,
    )
    plt.close(fig)

def save_step_prediction(
    img,
    images_dir,
    img_index,
    masks,
    matched_labels,
    dir_save_prediction_output,
    IHF,
    suffix,
    dpi=50,
):
    list_roi_matched = [l[img_index] for l in matched_labels]

    # confirmed labels
    draw_instructions = get_list_params_contours(list_roi_matched, masks, IHF, "confirmed")

    # plot image
    save_image_per_step(
        img, images_dir, 
        draw_instructions,
        dir_save_prediction_output,
        suffix,
        dpi,
    )