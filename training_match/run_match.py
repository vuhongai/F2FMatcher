import os, pickle, shutil, argparse, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
sys.path.append("/media/DATABRUT/DB_DDC/serverGPU/Cache_GPU_Ai/fiber_matcher/FFMatcher/training_match/")

from configs import *
from FM_match import *
from helper import * 

set_seed(random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parse arguments
parser = argparse.ArgumentParser(description="Run fiber mapping across slides.")
parser.add_argument("--py_param_file", type=str, help="Path to the config python (.py) file.")

args = parser.parse_args()
py_param_file = args.py_param_file

###### import parameters from the input python file
params_input = import_module_from_path(py_param_file)

# Directory of images
source_1 = params_input.source_1
source_2 = params_input.source_2

# File format of inputs:
czi_img1 = params_input.czi_img1
czi_img2 = params_input.czi_img2

# Channel index 
channel_index_img1 = params_input.channel_index_img1
channel_index_img2 = params_input.channel_index_img2

# Finetuned Cellpose model for each staining, more models found at:
CP_model_name_1 = params_input.CP_model_name_1
CP_model_name_2 = params_input.CP_model_name_2

# parameters of the image scan and export [fluorescence/brightfield, objective, scale_factor during image export]
param_ref = params_input.param_ref
param_img1 = params_input.param_img1
param_img2 = params_input.param_img2

# List of image pairs to perform fiber mapping
list_pair_images = params_input.list_pair_images

# Whether to save the illustration output comparing 2 imgs 
export_output_images = params_input.export_output_images
save_step_prediction = params_input.save_step_prediction

# Where to save the output results
dir_save_output = params_input.dir_save_output

# Create the directories
images_dir = f"{dir_save_output}/images" #dir of images used in training
dir_tempo = f"{dir_save_output}/tempo/" #dir to dump the temporary files
dir_embedding = f"{dir_save_output}/VAE_embed"
dir_save_npz = f"{dir_save_output}/npz_256"
dir_save_prediction_output = f"{dir_save_output}/prediction_output"
dir_save_cellpose_masks = f"{dir_save_output}/out_CP_masks"

for d in [dir_save_output, images_dir, dir_tempo, dir_embedding, dir_save_npz, dir_save_prediction_output, dir_save_cellpose_masks]:
    os.makedirs(d, exist_ok=True)

# IHF or not
IHF1 = param_img1[0]=="fluorescence"
IHF2 = param_img2[0]=="fluorescence"

################################################################################################################
################################################################################################################
################################################################################################################
# PREDICTION

for img1, img2 in list_pair_images:
    print(f"Processing {img1} and {img2}")
    
    # copy the image files
    print("Load, Resize, and Copy input images in the local folder (temporarily)")
    if os.path.exists(f"{images_dir}/{img1}.png"):
        print(f"{img1} already exists in {images_dir}, skip copying")
    else:
        if czi_img1:
            import_resize_export_czi_file(
                find_img_path(img1, source_1),
                IHF1, channel_index_img1,
                images_dir,
                param_img1, 
                param_ref
            )
        else:
            resize_image(find_img_path(img1, source_1), images_dir, param_img1, param_ref)

    if os.path.exists(f"{images_dir}/{img2}.png"):
        print(f"{img2} already exists in {images_dir}, skip copying")
    else:
        if czi_img2:
            import_resize_export_czi_file(
                find_img_path(img2, source_2),
                IHF2, channel_index_img2,
                images_dir,
                param_img2, 
                param_ref
            )
        else:
            resize_image(find_img_path(img2, source_2), images_dir, param_img2, param_ref)

    # prepare the npz, npy inputs
    print("Prepare inputs for prediction")
    cp_output_1 = FM_generate_VAE_inputs(
        img1,
        images_dir,
        dir_save_npz,
        CP_model_name_1,
        dir_save_cellpose_masks
    )
    cp_output_2 = FM_generate_VAE_inputs(
        img2,
        images_dir,
        dir_save_npz,
        CP_model_name_2,
        dir_save_cellpose_masks
    )
    FM_generate_embedding(
        dir_save_npz,
        dir_embedding,
        VAE_checkpoint,
        device
    )
    
    list_label_1 = sorted([int(f.split("_roi_")[1].split(".")[0]) for f in os.listdir(dir_save_npz) if img1 in f])
    list_label_2 = sorted([int(f.split("_roi_")[1].split(".")[0]) for f in os.listdir(dir_save_npz) if img2 in f])
    print(f"n_ROI_img1={len(list_label_1)}, n_ROI_img2={len(list_label_2)}")
    
    label2index = {
        "img1": {l:i for i,l in enumerate(list_label_1)},
        "img2": {l:i for i,l in enumerate(list_label_2)},
    }
    
    index2label = {
        "img1": {i:l for i,l in enumerate(list_label_1)},
        "img2": {i:l for i,l in enumerate(list_label_2)},
    }

    # prediction
    dir_prediction_single_pair = f"{dir_save_prediction_output}/{img1}___vs___{img2}"
    os.makedirs(dir_prediction_single_pair, exist_ok=True)

    print("Start prediction")
    matched_labels_f, scores, spatial_dist, cp_outputs = match_fibers(
        img1, list_label_1, cp_output_1,
        img2, list_label_2, cp_output_2,
        label2index,
        dir_embedding,
        images_dir, 
        dir_save_cellpose_masks,
        cls_checkpoint,
        device,
        list_k = [3,5,7],
    
        # initial guess
        n_initial_guess = 80,
        n_pair_selected = 4,
        min_cls_logit_init = 0.75,
    
        # local prediction
        distance_neighbors_ref = 200,
        max_distance_affine = 150,
        max_cost_geo_neighbors_sides = 30,
        max_cost_geo_neighbors_angles = 0.15,
        min_cls_logit = 0.5,
        patience_label = 5,    
        n_neighbors_validation = 3,
        n_processes = 60,
        n_try_unannotated = 1,
    
        # Cellpose model
        CP_model_name_1 = CP_model_name_1,
        CP_model_name_2 = CP_model_name_2,

        # Save predicted pairs in each step for visualization
        save_step_prediction=save_step_prediction,
        dir_save_prediction_output=dir_prediction_single_pair,
    ) 

    print('Save prediction')
    with open(f"{dir_prediction_single_pair}/paired_labels.pkl", "wb") as f:
        pickle.dump(matched_labels_f, f)

    if export_output_images:
        print('Save prediction illustration \n')
        cp_output1, cp_output2 = cp_outputs
        labels1_filtered = filter_ROIs(cp_output1, size_crop)
        labels2_filtered = filter_ROIs(cp_output2, size_crop)
        
        o1 = [cp_output1, labels1_filtered]
        o2 = [cp_output2, labels2_filtered]
        
        save_FM_prediction(
            img1,
            images_dir,
            0, #0/1: 1st/2nd image in the pair
            o1,
            matched_labels_f,
            dir_prediction_single_pair,
            dpi=80,
            IHF=IHF1
        )
        
        save_FM_prediction(
            img2,
            images_dir,
            1, #0/1: 1st/2nd image in the pair
            o2,
            matched_labels_f,
            dir_prediction_single_pair,
            dpi=80,
            IHF=IHF2
        )
    
    os.system(f"rm {dir_embedding}/*")
    os.system(f"rm {dir_save_npz}/*")
    # os.system(f"rm {images_dir}/*")


