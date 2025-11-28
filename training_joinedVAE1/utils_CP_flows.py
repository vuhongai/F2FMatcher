from configs import *
import os, shutil
from cellpose import core, utils, models, metrics, models
import cellpose
import argparse
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import regionprops, label, find_contours
from skimage.morphology import remove_small_objects
from skimage import transform
import skimage
from skimage import io
from scipy.optimize import linear_sum_assignment
import cv2
import random
random.seed(random_seed)
from scipy.ndimage import distance_transform_edt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import zoom


use_GPU = True
savedir = "./"
model_path = f"/home/ddc/CP_model_zoo/models/{Cellpose_model_name}"
channels = [0, 0]
save_outline_image = False

cellprob_threshold = 0
flow_threshold = 0.4

model = models.CellposeModel(gpu=True, pretrained_model=model_path)
diameter = model.diam_labels

def segment_image(img_path):
    images = [cellpose.io.imread(img_path)]
    masks, flows, styles = model.eval(images, batch_size=64,
                                      channels=channels,
                                      diameter=diameter,
                                      flow_threshold=flow_threshold,
                                      cellprob_threshold=cellprob_threshold,
                                      progress=True,
                                      )
    # output from cellpose 2.2.2
    """
    Returns
    -------
    masks: list of 2D arrays, or single 3D array (if do_3D=True)
            labelled image, where 0=no masks; 1,2,...=mask labels

    flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
        flows[k][0] = XY flow in HSV 0-255
        flows[k][1] = XY flows at each pixel
        flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
        flows[k][3] = final pixel locations after Euler integration 

    styles: list of 1D arrays of length 256, or single 1D array (if do_3D=True)
        style vector summarizing each image, also used to estimate size of objects in image

    diams: list of diameters, or float (if do_3D=True)
    """
    masks = masks[0]
    props = regionprops(masks) #regionprops(label(masks)) - check the training dataset - which use label(masks)
    
    # reconstructed images with all CP outputs
    flows_x = flows[0][1][0]
    flows_y = flows[0][1][1]
    cell_prob = flows[0][2]
    img_rec = np.concatenate([flows_x[:,:,np.newaxis], flows_y[:,:,np.newaxis], cell_prob[:,:,np.newaxis]],
                        axis=-1
                       )
    return masks, props, img_rec

def filter_ROIs(cp_output, size_crop=size_crop):
    masks, props, _ = cp_output
    y_full, x_full = masks.shape
    half_size = int(size_crop // 2)

    ROIs = []
    for region in props:
        area = region.area
        cy, cx = region.centroid
    
        y_min = int(cy - half_size)
        y_max = y_min + size_crop
        x_min = int(cx - half_size)
        x_max = x_min + size_crop
    
        conds = int((y_min >= 0) * (y_max <= y_full) * (x_min >= 0) * (x_max <= x_full) * (area >= thrs_roi_area))
        if conds == 1:
            ROIs.append(region.label)
    return ROIs

def crop_stack_mask_original(
    cp_output, 
    label_id, 
    size_crop=size_crop,
    thrs_roi_area=thrs_roi_area,
):
    masks, props, img_rec = cp_output
    y_full, x_full = masks.shape
    half_size = int(size_crop // 2)

    # find region with matching label
    region = next((p for p in props if p.label == label_id), None)
    if region is None:
        return None

    area = region.area
    cy, cx = region.centroid

    y_min = int(cy - half_size)
    y_max = y_min + size_crop
    x_min = int(cx - half_size)
    x_max = x_min + size_crop

    conds = int((y_min >= 0) * (y_max <= y_full) * (x_min >= 0) * (x_max <= x_full) * (area >= thrs_roi_area))
    if conds == 1:
        mask_i = (masks == label_id).astype(np.uint8)
        x1 = img_rec[y_min:y_max, x_min:x_max, :]
        x2 = mask_i[y_min:y_max, x_min:x_max][..., np.newaxis]
        return x1, x2
    else:
        return None

def augmentation(
    max_rotation_deg=30,
    max_shear_deg=5,
    max_scale_dev=0.1,
):
    # Create same transform
    angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)
    shear = np.random.uniform(-max_shear_deg, max_shear_deg)
    scale = 1 + np.random.uniform(-max_scale_dev, max_scale_dev)
    tform = transform.AffineTransform(
        rotation=np.deg2rad(angle),
        shear=np.deg2rad(shear), 
        scale=(scale, scale)
    )
    return tform

def augment_whole_slide(
    cp_output,
    tform
):
    masks, props, img_rec = cp_output
    # Augment full mask
    masks_aug = transform.warp(
        masks, 
        tform.inverse, order=0, preserve_range=True, mode='constant'
    ).astype(np.uint16)
    masks_aug = label(masks_aug)

    # Augment img_rec
    img_rec_aug = np.stack([
        transform.warp(
            img_rec[..., i], 
            tform.inverse, order=1, preserve_range=True, mode='constant'
        ) \
        for i in range(img_rec.shape[-1])
    ], axis=-1).astype(img_rec.dtype)

    # ---- ADD RANDOM 90° ROTATION ----
    k = np.random.choice([0, 1, 2, 3])  # corresponds to 0°, 90°, 180°, 270°
    if k > 0:
        img_rec_aug = np.rot90(img_rec_aug, k=k, axes=(0, 1)).copy()
        masks_aug = np.rot90(masks_aug, k=k, axes=(0, 1)).copy()

    return (masks_aug, regionprops(masks_aug), img_rec_aug), k
    
def augment_ROI(
    cp_output,
    cp_output_aug, 
    tform, # rotation + distortion
    k, # RANDOM 90° ROTATION
    label_id
):
    masks, props, img_rec = cp_output
    masks_aug, props_aug, img_rec_aug = cp_output_aug
    mask_roi = (masks == label_id).astype(np.uint8)
    
    # Augment ROI mask
    roi_aug = transform.warp(
        mask_roi, 
        tform.inverse, order=0, preserve_range=True, mode='constant'
    ).astype(np.uint8)
    roi_aug = label(roi_aug)

    # Extract new ROI label after augmentation
    props_roi_aug = regionprops(roi_aug)
    if len(props_roi_aug) == 0:
        return None  # lost ROI during augmentation
        
    # Assume largest object corresponds to augmented ROI
    largest_region = max(props_roi_aug, key=lambda r: r.area)
    new_mask = roi_aug == largest_region.label

    # ---- ADD RANDOM 90° ROTATION ----
    new_mask = np.rot90(new_mask, k=k, axes=(0, 1)).copy()
    return new_mask

def augment_image_track_roi(
    cp_output,
    label_id,
    max_rotation_deg=30,
    max_shear_deg=5,
    max_scale_dev=0.1
):
    masks, props, img_rec = cp_output
    mask_roi = (masks == label_id).astype(np.uint8)

    # Create same transform
    angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)
    shear = np.random.uniform(-max_shear_deg, max_shear_deg)
    scale = 1 + np.random.uniform(-max_scale_dev, max_scale_dev)
    tform = transform.AffineTransform(
        rotation=np.deg2rad(angle),
        shear=np.deg2rad(shear), 
        scale=(scale, scale)
    )

    # Augment full mask
    masks_aug = transform.warp(
        masks, 
        tform.inverse, order=0, preserve_range=True, mode='constant'
    ).astype(np.uint16)
    masks_aug = label(masks_aug)

    # Augment ROI mask
    roi_aug = transform.warp(
        mask_roi, 
        tform.inverse, order=0, preserve_range=True, mode='constant'
    ).astype(np.uint8)
    roi_aug = label(roi_aug)

    # Extract new ROI label after augmentation
    props_roi_aug = regionprops(roi_aug)
    if len(props_roi_aug) == 0:
        return None  # lost ROI during augmentation
        
    # Assume largest object corresponds to augmented ROI
    largest_region = max(props_roi_aug, key=lambda r: r.area)
    new_mask = roi_aug == largest_region.label

    # Augment img_rec
    img_rec_aug = np.stack([
        transform.warp(
            img_rec[..., i], 
            tform.inverse, order=1, preserve_range=True, mode='constant'
        ) \
        for i in range(img_rec.shape[-1])
    ], axis=-1).astype(img_rec.dtype)

    # ---- ADD RANDOM 90° ROTATION ----
    k = np.random.choice([0, 1, 2, 3])  # corresponds to 0°, 90°, 180°, 270°
    if k > 0:
        img_rec_aug = np.rot90(img_rec_aug, k=k, axes=(0, 1)).copy()
        masks_aug = np.rot90(masks_aug, k=k, axes=(0, 1)).copy()
        new_mask = np.rot90(new_mask, k=k, axes=(0, 1)).copy()

    return (masks_aug, regionprops(masks_aug), img_rec_aug, new_mask)


def crop_stack_augmented_mask(
    img_rec, mask_i, 
    size_crop=size_crop, 
    thrs_roi_area=thrs_roi_area
):
    y_full, x_full = mask_i.shape
    half_size = size_crop // 2

    props = regionprops(label(mask_i))
    if len(props) == 0:
        return None
    region = props[0]  # only one region

    cy, cx = region.centroid
    area = region.area

    y_min = int(cy - half_size)
    y_max = y_min + size_crop
    x_min = int(cx - half_size)
    x_max = x_min + size_crop

    conds = int((y_min >= 0) * (y_max <= y_full) * (x_min >= 0) * (x_max <= x_full) * (area >= thrs_roi_area))
    if conds == 1:
        x1 = img_rec[y_min:y_max, x_min:x_max, :]
        x2 = mask_i[y_min:y_max, x_min:x_max][..., np.newaxis]
        return x1, x2
    else:
        return None

def plot_input(x1,x2):
    _, axs = plt.subplots(1, 2, figsize=(8,4))
    axs[0].imshow(x1)
    axs[0].axis('off')
    axs[1].imshow(x2)
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

def plot_mask_with_labels(mask):
    fig, ax = plt.subplots(figsize=(mask.shape[1] / 100, mask.shape[0] / 100), dpi=100)
    props = regionprops(label(mask))
    
    # Create a black background
    ax.imshow(np.zeros_like(mask), cmap='gray')
    
    # Plot contours
    for region in props:
        # Outline the region
        contours = find_contours(mask == region.label, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)
    
        # Plot the number
        y, x = region.centroid
        ax.text(x, y, str(region.label), color='white', fontsize=8,
                ha='center', va='center', fontweight='bold')
    
    ax.axis('off')
    plt.show()

def plot_mask_on_image(image_path, mask):
    """
    Overlay mask contours (red) and labels (white) on the original image
    given its path.
    """
    image = io.imread(image_path)

    # Ensure mask is labeled
    if mask.max() == 1 or mask.min() == 0:
        mask = label(mask)

    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)

    # Show the original image
    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    # Draw contours and labels
    for region in regionprops(mask):
        contours = find_contours(mask == region.label, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

        y, x = region.centroid
        ax.text(x, y, str(region.label), color='white', fontsize=8,
                ha='center', va='center', fontweight='bold')
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

def generate_X(cp_output, label_id, n_augmentation=5):
    masks, props, img_rec = cp_output
    Xs = []
    x = crop_stack_mask_original(cp_output, label_id)
    if x is not None:
        Xs.append(x)

    for _ in range(n_augmentation):
        aug_result = augment_image_track_roi(cp_output, label_id)
        if aug_result is not None:
            masks_aug, props_aug, img_rec_aug, mask_roi_aug = aug_result
            x_a = crop_stack_augmented_mask(img_rec_aug, mask_roi_aug)
            if x_a is not None:
                Xs.append(x_a)
    return Xs

def compare_2_slides(
    path_img1,
    path_img2,
    plot_partial=False, 
    df_pair = None,
    dpi=100
):
    mask1, p1, _ = segment_image(path_img1)
    mask2, p2, _ = segment_image(path_img2)

    image1 = io.imread(path_img1)
    image2 = io.imread(path_img2)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Proportional widths
    total_width = w1 + w2
    fig_width = total_width / dpi
    fig_height = max(h1, h2) / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = GridSpec(1, 2, width_ratios=[w1, w2])

    if plot_partial:
        df_0 = df_pair[
            (df_pair.Image1==path_img1.split("/")[-1].split(".")[0]) & 
            (df_pair.Image2==path_img2.split("/")[-1].split(".")[0]) &
            (~df_pair.ROI_I2.isna())
        ]
        df_0.ROI_I2 = [int(i) for i in df_0.ROI_I2.tolist()]
        list_roi1 = df_0.ROI_I1.tolist()
        list_roi2 = df_0.ROI_I2.tolist()
        assert len(list_roi1) == len(list(set(list_roi1)))
        assert len(list_roi2) == len(list(set(list_roi2)))

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
            if plot_partial:
                if (ax_idx==0) & (region.label in list_roi1):
                    ax.text(x, y, str(region.label), color='gray', fontsize=6,
                            ha='center', va='center',)

                elif (ax_idx==1) & (region.label in list_roi2):
                    ax.text(x, y, str(region.label), color='gray', fontsize=6,
                            ha='center', va='center', )

                else:
                    ax.text(x, y, str(region.label), color='white', fontsize=11,
                        ha='center', va='center', fontweight='bold')
            else:
                ax.text(x, y, str(region.label), color='white', fontsize=9,
                        ha='center', va='center', fontweight='bold')

        ax.set_aspect('equal')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def resize_image_zoom(image_np, size=(size_crop_resize, size_crop_resize)):
    """
    Resize a float image in [0, 1] using bilinear interpolation.
    """
    zoom_factors = (size[0] / image_np.shape[0], size[1] / image_np.shape[1], 1)
    return zoom(image_np, zoom_factors, order=1).astype(np.float32)

def generate_dataset_single_image(image_path, thrs_roi_area):
    cp_output = segment_image(image_path)
    pass