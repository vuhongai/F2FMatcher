# INPUTs for running fiber matching

# Directory of images, can be any format, as it will be converted to PNG later on
source_1 = "/media/DATABRUT/DB_DDC/serverGPU/Cache_GPU_Ai/fiber_matcher/test_images"
source_2 = "/media/DATABRUT/DB_DDC/serverGPU/Cache_GPU_Ai/fiber_matcher/test_images"

# File format of inputs:
czi_img1 = True
czi_img2 = True

# Channel index to compare (only for czi files of IHF images)
# Note: channel 1 is index 0, channel 2 is index 1, ...
channel_index_img1 = 1
channel_index_img2 = None

# Finetuned Cellpose model for each staining, more models found at:
## TA (COX/SDH/NADH) 10X: "CP_AV_TA_COX-SDH-NADH_AxioScan10X"
## TA/Qua/Dia (WGA) 10X: "CP_AV_WGA_Dia_Qua_TA_AxioScan10X"
## TA (RedSirius) 10X: "AV_CP_10X_TA_RedSirius"

CP_model_name_1="CP_AV_WGA_Dia_Qua_TA_AxioScan10X" #work well one Laminin of WT
CP_model_name_2= "CP_AV_TA_COX-SDH-NADH_AxioScan10X"

# parameters of the image scan and export [fluorescence/brightfield, objective, scale_factor during image export]
param_ref = ["fluorescence", "10X", 1.0] # reference
param_img1 = ["fluorescence", "10X", 1.0]
param_img2 = ["brightfield", "10X", 1.0]

# List of image pairs to perform fiber mapping
list_pair_images = [
    ("22-082_10X_DAPI_LAM_DYS_COL4_1-Scene-3-TAG03", "22-082_10X_NADH_9-Scene-3-TAG03"),
]

# Whether to save the illustration output comparing 2 imgs (but it take about 20-30 mins per pair)
export_output_images = True
save_step_prediction = True

# Where to save the output results
dir_save_output = "/DATA/fiber_mapping_DDC/AJ"