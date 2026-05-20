import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import disk
from scipy.stats import skew, kurtosis
from skimage.measure import regionprops

import czifile

from .config import FEATURE_STATISTICS, COMPARTMENTS


def to_uint8(x):
    x = x.astype(np.float32)
    return (x / 65535 * 255).astype(np.uint8)


def load_mask(mask_path):
    with open(mask_path, "rb") as f:
        masks_list = pickle.load(f)
    masks = masks_list[0]
    return masks


def load_czi_channel(czi_path, channel_idx):
    with czifile.CziFile(czi_path) as czi:
        arr = czi.asarray()
        arr = np.squeeze(arr)
        ch = arr[channel_idx]
        if arr.dtype == np.uint16:
            ch8 = to_uint8(ch)
        else:
            ch8 = ch.astype(np.uint8)
    return ch8


def load_czi_rgb(czi_path):
    with czifile.CziFile(czi_path) as czi:
        arr = czi.asarray()
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            if arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype == np.uint16:
            img8 = to_uint8(arr)
        else:
            img8 = arr.astype(np.uint8)
    return img8


def compute_features(pixels):
    stats = {}
    stats["mean"] = float(np.mean(pixels))
    stats["std"] = float(np.std(pixels))
    stats["p10"] = float(np.percentile(pixels, 10))
    stats["p25"] = float(np.percentile(pixels, 25))
    stats["p50"] = float(np.median(pixels))
    stats["p75"] = float(np.percentile(pixels, 75))
    stats["p90"] = float(np.percentile(pixels, 90))
    if len(pixels) > 3:
        stats["skew"] = float(skew(pixels))
        stats["kurt"] = float(kurtosis(pixels))
    else:
        stats["skew"] = np.nan
        stats["kurt"] = np.nan
    return stats


def extract_compartment_masks(mask_binary, dilation_rad, erosion_rad):
    whole = mask_binary.astype(bool)
    if erosion_rad > 0:
        cytoplasm = binary_erosion(whole, structure=disk(erosion_rad))
    else:
        cytoplasm = whole.copy()
    if dilation_rad > 0 or erosion_rad > 0:
        dilated = binary_dilation(whole, structure=disk(dilation_rad))
        membrane = dilated & ~cytoplasm
    else:
        membrane = np.zeros_like(whole, dtype=bool)
    return {"whole": whole, "membrane": membrane, "cytoplasm": cytoplasm}


def quantify_fiber(image_2d, mask_binary, dilation_rad, erosion_rad, prefix=""):
    comp_masks = extract_compartment_masks(mask_binary, dilation_rad, erosion_rad)
    result = {}
    for comp_name, comp_mask in comp_masks.items():
        pix = image_2d[comp_mask]
        if len(pix) == 0:
            for s in FEATURE_STATISTICS:
                result[f"{prefix}{comp_name}_{s}"] = np.nan
        else:
            stats = compute_features(pix)
            for s in FEATURE_STATISTICS:
                result[f"{prefix}{comp_name}_{s}"] = stats[s]
    area = int(np.sum(mask_binary))
    result[f"{prefix}area"] = area
    return result


def quantify_image_singlechannel(image_2d, masks, dilation_rad, erosion_rad, fiber_labels=None):
    props = regionprops(masks)
    if fiber_labels is None:
        fiber_labels = sorted(set(masks.ravel()) - {0})
    rows = []
    for label in fiber_labels:
        mask_i = (masks == label).astype(np.uint8)
        bbox = props[[p.label for p in props].index(label)].bbox
        y_min, x_min, y_max, x_max = bbox
        y_min = max(0, y_min - dilation_rad)
        y_max = min(masks.shape[0], y_max + dilation_rad)
        x_min = max(0, x_min - dilation_rad)
        x_max = min(masks.shape[1], x_max + dilation_rad)
        mask_i_cropped = mask_i[y_min:y_max, x_min:x_max]
        image_i = image_2d[y_min:y_max, x_min:x_max]
        feats = quantify_fiber(image_i, mask_i_cropped, dilation_rad, erosion_rad)
        feats["fiber_label"] = label
        rows.append(feats)
    return pd.DataFrame(rows)


def quantify_image_rgb(image_rgb, masks, dilation_rad, erosion_rad, fiber_labels=None):
    props = regionprops(masks)
    if fiber_labels is None:
        fiber_labels = sorted(set(masks.ravel()) - {0})
    rows = []
    for label in fiber_labels:
        mask_i = (masks == label).astype(np.uint8)
        bbox = props[[p.label for p in props].index(label)].bbox
        y_min, x_min, y_max, x_max = bbox
        y_min = max(0, y_min - dilation_rad)
        y_max = min(masks.shape[0], y_max + dilation_rad)
        x_min = max(0, x_min - dilation_rad)
        x_max = min(masks.shape[1], x_max + dilation_rad)
        mask_i_cropped = mask_i[y_min:y_max, x_min:x_max]
        row = {"fiber_label": label}
        for ch_idx, ch_name in enumerate(["R", "G", "B"]):
            image_i = image_rgb[y_min:y_max, x_min:x_max, ch_idx]
            feats = quantify_fiber(image_i, mask_i_cropped, dilation_rad, erosion_rad, prefix=f"{ch_name}_")
            row.update(feats)
        rows.append(row)
    return pd.DataFrame(rows)


def quantify_staining(
    sample_name,
    staining_name,
    cfg,
    czi_base_dir,
    cp_masks_dir,
    output_dir,
):
    scfg = cfg[staining_name]
    czi_dir = scfg["czi_dir"]
    czi_full_dir = os.path.join(czi_base_dir, czi_dir)
    czi_files = os.listdir(czi_full_dir)
    czi_file = [f for f in czi_files if sample_name.upper() in f.upper()]
    if len(czi_file) == 0:
        print(f"  No CZI found for {sample_name} - {staining_name}")
        return None
    czi_path = os.path.join(czi_full_dir, czi_file[0])
    base_name = os.path.splitext(czi_file[0])[0]
    mask_path = os.path.join(cp_masks_dir, f"{base_name}_CP_masks.pkl")
    if not os.path.exists(mask_path):
        print(f"  Mask not found for {base_name} - {staining_name}")
        return None
    masks = load_mask(mask_path)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{sample_name}_{staining_name}_quant.csv")
    if os.path.exists(out_path):
        print(f"  Already done: {sample_name} - {staining_name}")
        return pd.read_csv(out_path)
    dilation = scfg["dilation"]
    erosion = scfg["erosion"]
    if scfg["ihf"]:
        image = load_czi_channel(czi_path, scfg["channel"])
        df = quantify_image_singlechannel(
            image, masks, dilation, erosion
        )
    else:
        image_rgb = load_czi_rgb(czi_path)
        df = quantify_image_rgb(
            image_rgb, masks, dilation, erosion
        )
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df)} fibers)")
    return df
