configs = {
    "random_seed": 1024,
    # dataset preparation
    "thrs_roi_area": 100,
    "size_crop": 256,
    "n_ROI_per_image": 500,

    # traning
    "size_latent": 256,
    "size_crop_resize": 128,

    
    "negative_fold": 1, # ratio of negative/positive pair during training
    "Cellpose_model_name": "CP_AV_WGA_Dia_Qua_TA_AxioScan10X",
    "checkpoint_dir": "/DATA/fiber_matcher/model/",
    "dir_dataset": "/DATA/fiber_matcher/datasets",
    "images_dir": "/DATA/fiber_matcher/images/", #dir of images used in training
    "dir_tempo": "/DATA/fiber_matcher/tempo/", #dir to dump the temporary files
    "dir_uncropped": "/media/DATABRUT/DB_DDC/serverGPU/Cache_GPU_Ai/fiber_matcher/training_tif",
    "batch_size": 32,
}

for k,v in configs.items():
    globals()[k] = v