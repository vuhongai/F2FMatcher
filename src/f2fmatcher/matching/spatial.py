import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error
from itertools import combinations
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed

from f2fmatcher.segmentation.cellpose_seg import filter_ROIs


def spatial_signature(cp_output, list_labels, k=5):
    masks, props, img_rec = cp_output
    centroids = np.array([region.centroid for region in props])
    all_labels = np.array([region.label for region in props])
    label_to_index = {lbl: i for i, lbl in enumerate(all_labels)}
    D = cdist(centroids, centroids, metric="euclidean")

    distance_matrix = np.zeros((len(list_labels), k), dtype=np.float32)
    neighbors_dict = {}

    for i, label in enumerate(list_labels):
        idx = label_to_index[label]
        distances = D[idx].copy()
        distances[idx] = np.inf
        nearest = np.argsort(distances)[:k]
        neighbors_dict[label] = [all_labels[j] for j in nearest]
        distance_matrix[i] = distances[nearest]

    return neighbors_dict, distance_matrix


def inverse_l2_similarity(A, B):
    A_sq = np.sum(A ** 2, axis=1, keepdims=True)
    B_sq = np.sum(B ** 2, axis=1, keepdims=True).T
    dists = np.sqrt(np.maximum(A_sq + B_sq - 2 * np.dot(A, B.T), 1e-8))
    return 1 / (1 + dists)


def wasserstein_similarity_matrix_parallel(D1, D2, n_jobs=-1):
    n, m = len(D1), len(D2)

    def compute_row(i):
        return [1 / (1 + wasserstein_distance(D1[i], D2[j])) for j in range(m)]

    sim = Parallel(n_jobs=n_jobs)(
        delayed(compute_row)(i) for i in tqdm(range(n), desc="Wasserstein similarity")
    )
    return np.array(sim)


def get_centroid_dict(props, list_labels):
    return {region.label: np.array(region.centroid) for region in props if region.label in list_labels}


def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def angle(opposite, side1, side2):
    cos_val = np.clip((side1 ** 2 + side2 ** 2 - opposite ** 2) / (2 * side1 * side2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


def triangle_geometry(A, B, C):
    AB, BC, CA = dist(A, B), dist(B, C), dist(C, A)
    return np.array([AB, BC, CA]), np.array([angle(CA, AB, BC), angle(AB, BC, CA), angle(BC, CA, AB)])


def costs_geometry(sides_1, angles_1, sides_2, angles_2):
    cost_sides = mean_absolute_error(sides_1, sides_2)
    cost_angles = mean_absolute_error(angles_1 / 180, angles_2 / 180)
    return cost_sides, cost_angles


def get_neighbors_ref_by_distance(Di, idi_ref, list_label_i, distance_neighbors_ref):
    dist_ref_i = Di[idi_ref]
    neighbors_id = np.where(dist_ref_i < distance_neighbors_ref)[0]
    neighbors_label = [list_label_i[i] for i in neighbors_id]
    return neighbors_id, neighbors_label


def get_k_nearest(D, idx, k=3):
    distances = D[idx].copy()
    distances[idx] = np.inf
    return np.argsort(distances)[:k]
