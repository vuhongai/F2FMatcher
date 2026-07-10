import os
import numpy as np
import torch
from tqdm import tqdm

from f2fmatcher.vae.models import SharedMultiHeadVAE
from f2fmatcher.vae.dataset import DatasetForEmbeddingExtraction


def generate_embedding_dir(dir_save_npz, dir_embedding, VAE_checkpoint, device, img_names=None):
    vae = SharedMultiHeadVAE().to(device)
    vae.load_state_dict(torch.load(VAE_checkpoint, map_location=device))
    vae.eval()

    all_files = os.listdir(dir_save_npz)
    if img_names is not None:
        files = [f for f in all_files if any(n in f for n in img_names)]
    else:
        files = all_files

    list_path = [f"{dir_save_npz}/{f}" for f in files]
    ds = DatasetForEmbeddingExtraction(list_path)

    os.makedirs(dir_embedding, exist_ok=True)

    for i in tqdm(range(len(ds))):
        path = ds.files[i]
        base = os.path.basename(path).replace(".npz", ".npy")
        save_path = os.path.join(dir_embedding, base)
        if not os.path.exists(save_path):
            tensor = ds.create_input(path).unsqueeze(0).to(device)
            with torch.no_grad():
                mu = vae.encode(tensor).squeeze(0).cpu().numpy()
            np.save(save_path, mu)
