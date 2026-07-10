import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
from cellpose import models, io as cp_io
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects


def segment_image(img_path, CP_model_name, CP_model_path="/home/ddc/CP_model_zoo/models/",
                  savedir="./", channels=(0, 0), cellprob_threshold=0, flow_threshold=0.4):
    base_name = os.path.basename(img_path).split(".")[0]
    path_masks = f"{savedir}/{base_name}_CP_masks.pkl"
    path_flows = f"{savedir}/{base_name}_CP_flows.pkl"

    if os.path.exists(path_masks) and os.path.exists(path_flows):
        with open(path_masks, "rb") as f:
            masks = pickle.load(f)
        with open(path_flows, "rb") as f:
            flows = pickle.load(f)
    else:
        cp_model = models.CellposeModel(gpu=True, pretrained_model=f"{CP_model_path}/{CP_model_name}")
        diameter = cp_model.diam_labels
        images = [cp_io.imread(img_path)]
        masks, flows, styles = cp_model.eval(
            images, batch_size=64, channels=channels,
            diameter=diameter, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, progress=True,
        )
        with open(path_masks, "wb") as f:
            pickle.dump(masks, f)
        with open(path_flows, "wb") as f:
            pickle.dump(flows, f)
        _save_CP_mask_image(masks[0], base_name, savedir)

    masks = masks[0]
    props = regionprops(masks)
    flows_x = flows[0][1][0]
    flows_y = flows[0][1][1]
    cell_prob = flows[0][2]
    img_rec = np.stack([flows_x, flows_y, cell_prob], axis=-1)
    return masks, props, img_rec


def _save_CP_mask_image(masks, img_name, savedir):
    img = Image.fromarray(masks.astype(np.int32), mode="I")
    img.save(f"{savedir}/{img_name}_cp_masks.png")


def filter_ROIs(cp_output, size_crop=256, thrs_roi_area=100):
    masks, props, _ = cp_output
    y_full, x_full = masks.shape
    half = size_crop // 2

    valid_labels = []
    for region in props:
        cy, cx = region.centroid
        y_min = int(cy - half)
        y_max = y_min + size_crop
        x_min = int(cx - half)
        x_max = x_min + size_crop
        ok = (y_min >= 0) and (y_max <= y_full) and (x_min >= 0) and (x_max <= x_full) and (region.area >= thrs_roi_area)
        if ok:
            valid_labels.append(region.label)
    return valid_labels


def crop_stack_original(cp_output, label_id, size_crop=256, thrs_roi_area=100):
    masks, props, img_rec = cp_output
    y_full, x_full = masks.shape
    half = size_crop // 2

    region = next((p for p in props if p.label == label_id), None)
    if region is None:
        return None

    cy, cx = region.centroid
    y_min = int(cy - half)
    y_max = y_min + size_crop
    x_min = int(cx - half)
    x_max = x_min + size_crop

    if (y_min >= 0) and (y_max <= y_full) and (x_min >= 0) and (x_max <= x_full) and (region.area >= thrs_roi_area):
        mask_i = (masks == label_id).astype(np.uint8)
        x1 = img_rec[y_min:y_max, x_min:x_max, :]
        x2 = mask_i[y_min:y_max, x_min:x_max][..., np.newaxis]
        return x1, x2
    return None


def _save_single_vae_input(label_id, centroids_yx, masks, img_rec, img_name, dir_save_npz, size_crop):
    from f2fmatcher.segmentation.cellpose_seg import crop_stack_original
    cy, cx = centroids_yx[label_id]
    y_full, x_full = masks.shape
    half = size_crop // 2
    y_min = int(cy - half)
    y_max = y_min + size_crop
    x_min = int(cx - half)
    x_max = x_min + size_crop

    mask_i = (masks == label_id).astype(np.uint8)
    x1 = img_rec[y_min:y_max, x_min:x_max, :]
    x2 = mask_i[y_min:y_max, x_min:x_max]

    np.savez_compressed(
        f"{dir_save_npz}/{img_name}_roi_{label_id}.npz",
        flow_x=x1[:, :, 0],
        flow_y=x1[:, :, 1],
        cell_prob=x1[:, :, 2],
        roi_mask=x2,
    )


def generate_VAE_inputs(img_name, images_dir, dir_save_npz, CP_model_name,
                        dir_save_cellpose_masks, size_crop=256, CP_model_path=None, n_processes=60):
    image_path = f"{images_dir}/{img_name}.png"
    seg_kwargs = {}
    if CP_model_path is not None:
        seg_kwargs["CP_model_path"] = CP_model_path
    cp_output = segment_image(img_path=image_path, CP_model_name=CP_model_name,
                              savedir=dir_save_cellpose_masks, **seg_kwargs)
    masks, props, img_rec = cp_output
    labels = filter_ROIs(cp_output, size_crop)
    centroids_yx = {p.label: p.centroid for p in props if p.label in labels}

    Parallel(n_jobs=n_processes)(
        delayed(_save_single_vae_input)(lbl, centroids_yx, masks, img_rec, img_name, dir_save_npz, size_crop)
        for lbl in tqdm(labels, desc=f"Generate VAE inputs for {img_name}")
    )
    return cp_output


def get_CP_masks(img_path, CP_model_name=None, CP_model_path="/home/ddc/CP_model_zoo/models/",
                 savedir="./", channels=(0, 0), cellprob_threshold=0, flow_threshold=0.4):
    base_name = os.path.basename(img_path).split(".")[0]
    path_masks = f"{savedir}/{base_name}_CP_masks.pkl"

    if os.path.exists(path_masks):
        with open(path_masks, "rb") as f:
            masks = pickle.load(f)
    else:
        cp_model = models.CellposeModel(gpu=True, pretrained_model=f"{CP_model_path}/{CP_model_name}")
        diameter = cp_model.diam_labels
        images = [cp_io.imread(img_path)]
        masks, flows, styles = cp_model.eval(
            images, batch_size=64, channels=channels, diameter=diameter,
            flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, progress=True,
        )
        with open(path_masks, "wb") as f:
            pickle.dump(masks, f)
        _save_CP_mask_image(masks[0], base_name, savedir)

    masks = masks[0]
    props = regionprops(masks)
    return masks, props, base_name
