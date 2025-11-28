from helper import *

def import_resize_export_czi_RNAScope(
    czi_path,
    channel_index,
    thrsh_16bit,
    param_img,
    param_ref,
    dir_image_downstream,
):
    with czifile.CziFile(czi_path) as czi:
        arr = czi.asarray()
        arr = np.squeeze(arr)
        ch = arr[channel_index]
        ch = ((ch>thrsh_16bit) * 255).astype(np.uint8)
    
    # scale the image to the reference
    px_size_ref_um = pixel_sizes[param_ref[0]][param_ref[1]] * param_ref[2]
    px_size_img_um = pixel_sizes[param_img[0]][param_img[1]] # czi image: param_img[2] = 1.0
    scale_factor = px_size_img_um / px_size_ref_um
    
    img = Image.fromarray(ch)
    w, h = img.size
    w_resized = max(1, int(w*scale_factor))
    h_resized = max(1, int(h*scale_factor))
    img_resized = img.resize((w_resized, h_resized))
    
    base_name = os.path.basename(czi_path).split('.')[0]
    img_resized.save(f"{dir_image_downstream}/{base_name}_c{channel_index}.png", format="PNG")