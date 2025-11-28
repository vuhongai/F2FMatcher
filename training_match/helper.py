from PIL import Image
import os, sys
import czifile
import numpy as np
import importlib.util
from pathlib import Path

pixel_sizes = {
    "fluorescence": {"2.5X": 2.604, "10X": 0.651, "20X": 0.3255, "40X": 0.16275},
    "brightfield": {"2.5X": 1.76, "10X": 0.44, "20X": 0.22, "40X": 0.11},
}

def resize_image(
    path_img,
    dir_save_img,
    param_img = ["brightfield", "10X", 1.0],
    param_ref = ["fluorescence", "10X", 1.0],
):
    """
        Inputs:
        - path_img: path img to resize (IMPORTANT: img with any format other than czi)
        - param_ref/img: [fluorescence/brightfield, objective, scale_factor during image export]
        - dir_save_img: directory to save the resized image
    """
    px_size_ref_um = pixel_sizes[param_ref[0]][param_ref[1]] * param_ref[2]
    px_size_img_um = pixel_sizes[param_img[0]][param_img[1]] * param_img[2]
    scale_factor = px_size_img_um / px_size_ref_um
    base_name = os.path.basename(path_img).split('.')[0]
    path_out = f"{dir_save_img}/{base_name}.png"
    
    with Image.open(path_img) as img:
        w, h = img.size
        w_resized = max(1, int(w*scale_factor))
        h_resized = max(1, int(h*scale_factor))
        img_resized = img.resize((w_resized, h_resized))
        img_resized.save(path_out, format="PNG")

def find_img_path(img_name, dir_image):
    path_img = [p for p in os.listdir(dir_image) if f"{img_name}." in p]
    if len(path_img)==1:
        path_img = path_img[0]
    else:
        print(f"'Can't find the {img_name}")
    return os.path.join(dir_image, path_img)

def to_uint8(x):
    x = x.astype(np.float32)
    return (x / 65535 * 255).astype(np.uint8)

def import_resize_export_czi_file(
    czi_path,
    IHF,
    channel_index,
    dir_save_png,
    param_img = ["brightfield", "10X", 1.0],
    param_ref = ["fluorescence", "10X", 1.0],
):
    """
    Process czi file (from AxioScan)
    Input: 
    - czi_path: full path to czi file
    - IHF: whether it is immunofluoresence image or brightfield?
    - channel_index: channel of interest
    - dir_save_png: where to save the png output
    - param_ref/img: [fluorescence/brightfield, objective, scale_factor during image export] # scale_factor during image export is inactivated here.
    """
    with czifile.CziFile(czi_path) as czi:
        arr = czi.asarray()
        arr = np.squeeze(arr) # shape (Channels, Y, X)
    
        # convert to 8bit
        if IHF:
            # isolate the channel and convert to 8bit
            ch = arr[channel_index]
            ch8 = to_uint8(ch)
        else:
            if arr.ndim == 3 and arr.shape[0] == 3:     # (C, Y, X)
                arr = np.transpose(arr, (1, 2, 0))      # → (Y, X, C)
            ch8 = to_uint8(arr)
    
        # scale the image to the reference
        px_size_ref_um = pixel_sizes[param_ref[0]][param_ref[1]] * param_ref[2]
        px_size_img_um = pixel_sizes[param_img[0]][param_img[1]] # czi image: param_img[2] = 1.0
        scale_factor = px_size_img_um / px_size_ref_um
        img = Image.fromarray(ch8)
        w, h = img.size
        w_resized = max(1, int(w*scale_factor))
        h_resized = max(1, int(h*scale_factor))
        img_resized = img.resize((w_resized, h_resized))
        
        # save the image
        base_name = os.path.basename(czi_path).split('.')[0]
        img_resized.save(f"{dir_save_png}/{base_name}.png", format="PNG")

def import_module_from_path(path):
    path = Path(path).resolve()
    module_name = path.stem  # file name without .py

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module