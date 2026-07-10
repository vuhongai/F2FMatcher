from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import disk


def dilate_mask(mask_i, n_px):
    return binary_dilation(mask_i, structure=disk(n_px)).astype("uint8")


def erode_mask(mask_i, n_px):
    return binary_erosion(mask_i, structure=disk(n_px)).astype("uint8")


def isolate_membrane_mask(mask_i, n_px_erosion, n_px_dilation):
    dilated = binary_dilation(mask_i, structure=disk(n_px_dilation)).astype("uint8")
    eroded = binary_erosion(mask_i, structure=disk(n_px_erosion)).astype("uint8")
    return dilated - eroded


def extract_single_ROI_from_mask(masks, label_id, bbox_i, n_px_dilation=0, n_px_erosion=0):
    y_full, x_full = masks.shape
    y_min, x_min, y_max, x_max = bbox_i
    y_min = max(0, y_min - n_px_dilation)
    y_max = min(y_full, y_max + n_px_dilation)
    x_min = max(0, x_min - n_px_dilation)
    x_max = min(x_full, x_max + n_px_dilation)
    bbox_i = [y_min, x_min, y_max, x_max]

    mask_i = (masks == label_id).astype("uint8")
    mask_i = mask_i[y_min:y_max, x_min:x_max]
    mask_membrane = isolate_membrane_mask(mask_i, n_px_erosion, n_px_dilation)
    mask_cytoplasm = erode_mask(mask_i, n_px=n_px_erosion)
    return bbox_i, mask_i, mask_membrane, mask_cytoplasm


def crop_image_with_bbox(image, bbox_i):
    y_min, x_min, y_max, x_max = bbox_i
    return image[y_min:y_max, x_min:x_max]
