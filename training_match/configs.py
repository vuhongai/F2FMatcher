configs = {
    "random_seed": 1024,
    
    # dataset preparation
    "thrs_roi_area": 100,
    "size_crop": 256,

    # VAE model
    "size_latent": 256,
    "size_crop_resize": 128,
    "VAE_checkpoint": "/DATA/fiber_matcher/model/LatentVAE2_256_128.pth",

    # Classifier model
    "cls_checkpoint": "/DATA/fiber_matcher/model/fibermatcher_cls_2.pth",
    "negative_fold": 4, # ratio of negative/positive pair during training
    # augmentation parameters 
    "n_augmentation": 50,
    "max_rotation_deg": 90,
    "max_shear_deg": 5,
    "max_scale_dev": 0.1,
    
    # Cellpose model
    "Cellpose_model_name": "CP_AV_WGA_Dia_Qua_TA_AxioScan10X",
    
    # directories
    "images_dir": "/DATA/fiber_matcher/test/images", #dir of images used in training
    "dir_tempo": "/DATA/fiber_matcher/tempo/", #dir to dump the temporary files
    "dir_embedding": "/DATA/fiber_matcher/test/VAE_embed",
    "dir_save_npz": "/DATA/fiber_matcher/test/npz_256",
    "dir_save_prediction_output": "/DATA/fiber_matcher/test/prediction_output",
    "dir_save_cellpose_masks": "/DATA/fiber_matcher/test/out_CP_masks",

    # parameters for prediction
    "list_k": [3,5,7],
    "min_spatial_distance": 0,
    "min_cls_logit": 0.25,
    "min_cls_logit_init": 0.5,
    "n_initial_guess": 10,
    "n_neighbors": 3,
    "max_cost_geo_neighbors": 1,
    "patience_label": 10,
    "n_log_pairs": 50,

    "n_processes": 60,

}

for k,v in configs.items():
    globals()[k] = v