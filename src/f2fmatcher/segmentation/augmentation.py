import numpy as np
from skimage import transform
from skimage.measure import regionprops, label


def random_transform(max_rotation_deg=90, max_shear_deg=5, max_scale_dev=0.1):
    angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)
    shear = np.random.uniform(-max_shear_deg, max_shear_deg)
    scale = 1 + np.random.uniform(-max_scale_dev, max_scale_dev)
    return transform.AffineTransform(
        rotation=np.deg2rad(angle),
        shear=np.deg2rad(shear),
        scale=(scale, scale),
    )


def augment_whole_slide(cp_output, tform):
    masks, props, img_rec = cp_output
    masks_aug = transform.warp(masks, tform.inverse, order=0, preserve_range=True, mode="constant").astype(np.uint16)
    masks_aug = label(masks_aug)
    img_rec_aug = np.stack([
        transform.warp(img_rec[..., i], tform.inverse, order=1, preserve_range=True, mode="constant")
        for i in range(img_rec.shape[-1])
    ], axis=-1).astype(img_rec.dtype)

    k = np.random.choice([0, 1, 2, 3])
    if k > 0:
        img_rec_aug = np.rot90(img_rec_aug, k=k, axes=(0, 1)).copy()
        masks_aug = np.rot90(masks_aug, k=k, axes=(0, 1)).copy()

    return (masks_aug, regionprops(masks_aug), img_rec_aug), k


def augment_ROI(cp_output, cp_output_aug, tform, k, label_id):
    masks, props, img_rec = cp_output
    masks_aug, props_aug, img_rec_aug = cp_output_aug
    mask_roi = (masks == label_id).astype(np.uint8)

    roi_aug = transform.warp(mask_roi, tform.inverse, order=0, preserve_range=True, mode="constant").astype(np.uint8)
    roi_aug = label(roi_aug)
    props_roi = regionprops(roi_aug)
    if len(props_roi) == 0:
        return None
    largest = max(props_roi, key=lambda r: r.area)
    new_mask = (roi_aug == largest.label)
    new_mask = np.rot90(new_mask, k=k, axes=(0, 1)).copy()
    return new_mask


def augment_image_track_roi(cp_output, label_id, max_rotation_deg=30, max_shear_deg=5, max_scale_dev=0.1):
    masks, props, img_rec = cp_output
    mask_roi = (masks == label_id).astype(np.uint8)

    tform = random_transform(max_rotation_deg, max_shear_deg, max_scale_dev)

    masks_aug = transform.warp(masks, tform.inverse, order=0, preserve_range=True, mode="constant").astype(np.uint16)
    masks_aug = label(masks_aug)

    roi_aug = transform.warp(mask_roi, tform.inverse, order=0, preserve_range=True, mode="constant").astype(np.uint8)
    roi_aug = label(roi_aug)
    props_roi = regionprops(roi_aug)
    if len(props_roi) == 0:
        return None
    largest = max(props_roi, key=lambda r: r.area)
    new_mask = (roi_aug == largest.label)

    img_rec_aug = np.stack([
        transform.warp(img_rec[..., i], tform.inverse, order=1, preserve_range=True, mode="constant")
        for i in range(img_rec.shape[-1])
    ], axis=-1).astype(img_rec.dtype)

    k = np.random.choice([0, 1, 2, 3])
    if k > 0:
        img_rec_aug = np.rot90(img_rec_aug, k=k, axes=(0, 1)).copy()
        masks_aug = np.rot90(masks_aug, k=k, axes=(0, 1)).copy()
        new_mask = np.rot90(new_mask, k=k, axes=(0, 1)).copy()

    return masks_aug, regionprops(masks_aug), img_rec_aug, new_mask


def crop_stack_augmented_mask(img_rec, mask_i, size_crop=256, thrs_roi_area=100):
    props = regionprops(label(mask_i))
    if len(props) == 0:
        return None
    region = props[0]
    cy, cx = region.centroid
    half = size_crop // 2
    y_min = int(cy - half)
    y_max = y_min + size_crop
    x_min = int(cx - half)
    x_max = x_min + size_crop
    y_full, x_full = mask_i.shape

    if (y_min >= 0) and (y_max <= y_full) and (x_min >= 0) and (x_max <= x_full) and (region.area >= thrs_roi_area):
        x1 = img_rec[y_min:y_max, x_min:x_max, :]
        x2 = mask_i[y_min:y_max, x_min:x_max][..., np.newaxis]
        return x1, x2
    return None


def generate_X(cp_output, label_id, n_augmentation=5):
    from f2fmatcher.segmentation.cellpose_seg import crop_stack_original
    Xs = []
    x = crop_stack_original(cp_output, label_id)
    if x is not None:
        Xs.append(x)
    for _ in range(n_augmentation):
        result = augment_image_track_roi(cp_output, label_id)
        if result is not None:
            masks_aug, props_aug, img_rec_aug, mask_roi_aug = result
            x_a = crop_stack_augmented_mask(img_rec_aug, mask_roi_aug)
            if x_a is not None:
                Xs.append(x_a)
    return Xs
