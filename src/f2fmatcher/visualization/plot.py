import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import regionprops, find_contours, label
from skimage.morphology import remove_small_objects
from tqdm import tqdm
import multiprocessing

from f2fmatcher.segmentation.cellpose_seg import filter_ROIs


def save_FM_prediction(img_name, images_dir, img_index, cp_output_list,
                       matched_labels, save_dir, dpi=80, IHF=True):
    os.makedirs(save_dir, exist_ok=True)
    cp_output, labels_filtered = cp_output_list
    masks, props, img_rec = cp_output
    image_path = f"{images_dir}/{img_name}.png"

    from skimage import io
    image = io.imread(image_path)
    fig, ax = plt.subplots(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)

    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)

    color_contour = "red" if IHF else "gray"
    text_color_match = "white" if IHF else "black"
    text_color_valid = "red"
    text_color_ignored = "gray"

    for region in regionprops(masks):
        contours = find_contours(masks == region.label, 0.5)
        label_val = region.label

        if label_val in [p[img_index] for p in matched_labels]:
            txt = f"{label_val}-{dict(matched_labels)[label_val]}" if img_index == 0 else \
                  f"{dict((v, k) for k, v in dict(matched_labels).items())[label_val]}-{label_val}"
            color = text_color_match
            weight = "bold"
            fontsize = 10
        elif label_val in labels_filtered:
            txt = str(label_val)
            color = text_color_valid
            weight = "normal"
            fontsize = 10
        else:
            txt = str(label_val)
            color = text_color_ignored
            weight = "normal"
            fontsize = 8

        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color_contour, linewidth=1)

        y, x = region.centroid
        ax.text(x, y, txt, color=color, fontsize=fontsize, ha="center", va="center", fontweight=weight)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{img_name}_prediction.png", bbox_inches="tight", dpi=dpi)
    plt.close()


def compare_2_slides(path_img1, path_img2, dpi=100, size_crop=256,
                     CP_model_name_1=None, CP_model_name_2=None, dir_save_cellpose_masks="./",
                     matched_labels=None, plot_annotated_pairs=False):
    from f2fmatcher.segmentation.cellpose_seg import segment_image

    cp1 = segment_image(path_img1, CP_model_name_1, savedir=dir_save_cellpose_masks)
    cp2 = segment_image(path_img2, CP_model_name_2, savedir=dir_save_cellpose_masks)
    mask1, p1, _ = cp1
    mask2, p2, _ = cp2
    labels1 = filter_ROIs(cp1, size_crop)
    labels2 = filter_ROIs(cp2, size_crop)

    from skimage import io
    image1 = io.imread(path_img1)
    image2 = io.imread(path_img2)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    total_width = w1 + w2
    fig_width = total_width / dpi
    fig_height = max(h1, h2) / dpi

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = GridSpec(1, 2, width_ratios=[w1, w2])

    for i, (img, mask, ax_idx) in enumerate(zip([image1, image2], [mask1, mask2], [0, 1])):
        ax = fig.add_subplot(gs[ax_idx])
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

        for region in regionprops(mask):
            contours = find_contours(mask == region.label, 0.5)
            for c in contours:
                ax.plot(c[:, 1], c[:, 0], color="red", linewidth=1)
            y, x = region.centroid
            ax.text(x, y, str(region.label), color="white", fontsize=10,
                    ha="center", va="center", fontweight="bold")

        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


_params_step_viz = {
    "IHF": {
        "confirmed": {"color_contour": "red", "size_contour": 1, "facecolor": "white", "alpha": 0.5},
        "on-going": {"color_contour": "red", "size_contour": 0.5, "facecolor": "white", "alpha": 0.2},
        "ignored": {"color_contour": "black", "size_contour": 0.1, "facecolor": "white", "alpha": 0.0},
    },
    "BrighField": {
        "confirmed": {"color_contour": "black", "size_contour": 1, "facecolor": "red", "alpha": 0.5},
        "on-going": {"color_contour": "black", "size_contour": 0.5, "facecolor": "red", "alpha": 0.2},
        "ignored": {"color_contour": "white", "size_contour": 0.1, "facecolor": "red", "alpha": 0.0},
    },
}


def prepare_region_annotation(region_label, masks, matched_labels, labels_filtered, img_index, IHF):
    contours = find_contours(masks == region_label, 0.5)
    y, x = np.argwhere(masks == region_label).mean(axis=0)

    if region_label in [p[img_index] for p in matched_labels]:
        txt = f"{region_label}-{dict(matched_labels)[region_label]}" if img_index == 0 else \
              f"{dict((v, k) for k, v in dict(matched_labels).items())[region_label]}-{region_label}"
        color = "white" if IHF else "black"
        weight = "bold"
        fontsize = 10
    elif region_label in labels_filtered:
        txt, color, weight, fontsize = str(region_label), "red", "normal", 10
    else:
        txt, color, weight, fontsize = str(region_label), "gray", "normal", 8

    return {
        "contours": contours,
        "text": txt,
        "coords": (x, y),
        "color": color,
        "weight": weight,
        "fontsize": fontsize,
        "color_contour": "red" if IHF else "gray",
    }
