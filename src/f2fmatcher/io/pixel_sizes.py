PIXEL_SIZES = {
    "fluorescence": {"2.5X": 2.604, "5X": 1.302, "10X": 0.651, "20X": 0.3255, "40X": 0.16275},
    "brightfield": {"2.5X": 1.76, "5X": 0.88, "10X": 0.44, "20X": 0.22, "40X": 0.11},
}


def get_pixel_size(microscopy_type, objective, scale_factor=1.0):
    return PIXEL_SIZES[microscopy_type][objective] * scale_factor


def compute_resize_scale(param_img, param_ref):
    px_ref = get_pixel_size(param_ref[0], param_ref[1], param_ref[2])
    px_img = get_pixel_size(param_img[0], param_img[1], param_img[2])
    return px_img / px_ref
