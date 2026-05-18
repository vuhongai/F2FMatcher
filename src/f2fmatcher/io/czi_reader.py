import os
import numpy as np
from PIL import Image
import czifile

from f2fmatcher.io.pixel_sizes import compute_resize_scale


def to_uint8(x):
    x = x.astype(np.float32)
    return (x / 65535 * 255).astype(np.uint8)


def import_resize_export_czi(
    czi_path,
    IHF,
    channel_index,
    dir_save_png,
    param_img=("brightfield", "10X", 1.0),
    param_ref=("fluorescence", "10X", 1.0),
):
    with czifile.CziFile(czi_path) as czi:
        arr = czi.asarray()
        arr = np.squeeze(arr)
        unit16 = arr.dtype == np.uint16

        if IHF:
            ch = arr[channel_index]
            ch8 = to_uint8(ch) if unit16 else ch.astype(np.uint8)
        else:
            if arr.ndim == 3:
                if arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                elif arr.shape[-1] != 3:
                    raise ValueError("Unexpected shape for brightfield image.")
            else:
                raise ValueError("Unexpected shape for brightfield image.")
            ch8 = to_uint8(arr) if unit16 else arr.astype(np.uint8)

        scale = compute_resize_scale(param_img, param_ref)
        img = Image.fromarray(ch8)
        w, h = img.size
        w_resized = max(1, int(w * scale))
        h_resized = max(1, int(h * scale))
        img_resized = img.resize((w_resized, h_resized))

        base_name = os.path.basename(czi_path).split(".")[0]
        img_resized.save(f"{dir_save_png}/{base_name}.png", format="PNG")


def resize_image(path_img, dir_save_img, param_img=("fluorescence", "10X", 1.0), param_ref=("brightfield", "10X", 1.0)):
    scale = compute_resize_scale(param_img, param_ref)
    base_name = os.path.basename(path_img).split(".")[0]
    path_out = f"{dir_save_img}/{base_name}.png"

    with Image.open(path_img) as img:
        w, h = img.size
        w_resized = max(1, int(w * scale))
        h_resized = max(1, int(h * scale))
        img.resize((w_resized, h_resized)).save(path_out, format="PNG")


def find_img_path(img_name, dir_image):
    matches = [p for p in os.listdir(dir_image) if f"{img_name}." in p]
    if len(matches) == 1:
        return os.path.join(dir_image, matches[0])
    raise FileNotFoundError(f"Cannot find {img_name} in {dir_image}")


def save_singlechannel_from_czi(czi_path, dir_save_png, selected_channel):
    with czifile.CziFile(czi_path) as czi:
        arr = czi.asarray()
        arr = np.squeeze(arr)
        arr = arr[selected_channel]
        ch8 = to_uint8(arr)
        img = Image.fromarray(ch8)
        base_name = os.path.basename(czi_path).split(".")[0]
        img.save(f"{dir_save_png}/{base_name}_c{selected_channel}.png", format="PNG")


def save_multichannels_from_czi(czi_path, dir_save_png, selected_channels):
    with czifile.CziFile(czi_path) as czi:
        arr = czi.asarray()
        arr = np.squeeze(arr)
        arr = arr[selected_channels[::-1]]
        ch8 = to_uint8(arr)
        ch8 = np.transpose(ch8, (1, 2, 0))
        img = Image.fromarray(ch8)
        base_name = os.path.basename(czi_path).split(".")[0]
        img.save(f"{dir_save_png}/{base_name}.png", format="PNG")
