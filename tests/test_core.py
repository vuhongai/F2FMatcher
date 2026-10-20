import numpy as np
import torch

from f2fmatcher.vae.models import VAEEncoder, SharedMultiHeadVAE, count_parameters
from f2fmatcher.classifier.model import PairClassifier
from f2fmatcher.matching.spatial import spatial_signature, triangle_geometry, costs_geometry


def test_vae_encoder_output_shape():
    model = VAEEncoder(latent_dim=256)
    x = torch.randn(4, 3, 128, 128)
    mu, logvar = model(x)
    assert mu.shape == (4, 256), f"Expected (4, 256), got {mu.shape}"
    assert logvar.shape == (4, 256), f"Expected (4, 256), got {logvar.shape}"


def test_shared_vae_forward():
    model = SharedMultiHeadVAE(latent_dim=256)
    x = torch.randn(2, 3, 128, 128)
    fx, fy, mask, mu, logvar = model(x)
    assert fx.shape == (2, 1, 128, 128)
    assert fy.shape == (2, 1, 128, 128)
    assert mask.shape == (2, 1, 128, 128)
    assert mu.shape == (2, 256)


def test_vae_encode():
    model = SharedMultiHeadVAE(latent_dim=256)
    x = torch.randn(2, 3, 128, 128)
    mu = model.encode(x)
    assert mu.shape == (2, 256)


def test_vae_count_parameters():
    model = SharedMultiHeadVAE()
    n = count_parameters(model)
    assert n > 0, "Model should have parameters"


def test_classifier_output_shape():
    model = PairClassifier(embedding_dim=256)
    emb1 = torch.randn(8, 256)
    emb2 = torch.randn(8, 256)
    out = model(emb1, emb2)
    assert out.shape == (8, 1), f"Expected (8, 1), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Sigmoid output must be in [0,1]"


def test_spatial_signature():
    n_labels = 10
    fake_masks = np.zeros((100, 100), dtype=np.uint16)
    for i in range(1, n_labels + 1):
        fake_masks[i * 8:(i * 8 + 5), 20:25] = i

    from skimage.measure import regionprops
    props = regionprops(fake_masks)
    img_rec = np.zeros((100, 100, 3))
    cp_output = (fake_masks, props, img_rec)
    list_labels = list(range(1, n_labels + 1))

    neighbors, dist_matrix = spatial_signature(cp_output, list_labels, k=3)
    assert dist_matrix.shape == (n_labels, 3), f"Expected ({n_labels}, 3), got {dist_matrix.shape}"
    assert len(neighbors) == n_labels


def test_triangle_geometry():
    A, B, C = np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
    sides, angles = triangle_geometry(A, B, C)
    assert len(sides) == 3
    assert len(angles) == 3
    assert sides[0] == 1.0  # AB = 1
    assert sides[2] == 1.0  # CA = 1
    assert abs(sides[1] - np.sqrt(2)) < 1e-6  # BC = sqrt(2)


def test_costs_geometry():
    A1, B1, C1 = np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
    A2, B2, C2 = np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
    s1, a1 = triangle_geometry(A1, B1, C1)
    s2, a2 = triangle_geometry(A2, B2, C2)
    cs, ca = costs_geometry(s1, a1, s2, a2)
    assert cs == 0.0, "Identical triangles should have zero side cost"
    assert ca == 0.0, "Identical triangles should have zero angle cost"
