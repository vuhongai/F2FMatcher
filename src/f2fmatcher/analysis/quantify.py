import numpy as np
from skimage.measure import regionprops, label

from f2fmatcher.analysis.membrane import extract_single_ROI_from_mask


def quantify_staining_per_ROI(image, masks, n_px_dilation=0, n_px_erosion=0):
    props = regionprops(masks)
    results = {}
    for region in props:
        bbox, mask_i, mask_membrane, mask_cytoplasm = extract_single_ROI_from_mask(
            masks, region.label, region.bbox, n_px_dilation, n_px_erosion
        )
        roi_img = crop_image_with_bbox(image, bbox)
        results[region.label] = {
            "area": region.area,
            "centroid": region.centroid,
            "bbox": bbox,
            "mean_intensity_whole": roi_img[mask_i > 0].mean(),
            "mean_intensity_membrane": roi_img[mask_membrane > 0].mean() if mask_membrane.sum() > 0 else 0,
            "mean_intensity_cytoplasm": roi_img[mask_cytoplasm > 0].mean() if mask_cytoplasm.sum() > 0 else 0,
            "total_intensity": roi_img[mask_i > 0].sum(),
        }
    return results


def compile_quantification(image1, image2, masks1, matched_labels):
    q1 = quantify_staining_per_ROI(image1, masks1)
    q2 = quantify_staining_per_ROI(image2, masks2)

    rows = []
    for l1, l2 in matched_labels:
        rows.append({**q1[l1], "label_img1": l1, "label_img2": l2, **q2[l2]})
    return rows


def crop_image_with_bbox(image, bbox_i):
    y_min, x_min, y_max, x_max = bbox_i
    return image[y_min:y_max, x_min:x_max]
