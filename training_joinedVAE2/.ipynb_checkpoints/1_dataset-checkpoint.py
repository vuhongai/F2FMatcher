import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
import pickle
from tqdm import tqdm

from configs import *
from utils_CP_flows import *
import cellpose
from scipy.ndimage import zoom
import random
random.seed(random_seed)
from multiprocessing import Pool
from functools import partial


j = 0 # augmentation index

for file in tqdm(os.listdir(dir_uncropped)):
    image_path = f"{dir_uncropped}/{file}"
    img_name = image_path.split("/")[-1].split(".")[0]
    
    img = io.imread(image_path, as_gray=True,)
    
    ## randomly isolate ROI and its surrounding
    # segmentation by CP
    cp_output = segment_image(image_path)
    masks, props, img_rec = cp_output
    labels = filter_ROIs(cp_output, size_crop)

    # augment the full image
    tform = augmentation()
    cp_output_aug, k = augment_whole_slide(cp_output,tform)
    masks_aug, props_aug, img_rec_aug = cp_output_aug

    def process_single_roi(label_id):
        if not os.path.exists(f"{dir_dataset}/npz_{size_crop}/{img_name}_roi_{label_id}.npz"):
            # original crop
            x1, x2 = crop_stack_mask_original(cp_output, label_id)
            np.savez_compressed(
                f"{dir_dataset}/npz_{size_crop}/{img_name}_roi_{label_id}.npz",
                flow_x=x1[...,0],
                flow_y=x1[...,1],
                cell_prob=x1[...,2],
                roi_mask=x2[...,0]
            )
            
            # Augmented crop
            mask_roi_aug = augment_ROI(cp_output, cp_output_aug, tform, k, label_id)
            if mask_roi_aug is not None:
                x_a = crop_stack_augmented_mask(img_rec_aug, mask_roi_aug)
                if x_a is not None:
                    x1a, x2a = x_a
                    path_aug = f"{dir_dataset}/npz_{size_crop}_aug{j}/{img_name}_roi_{label_id}_aug{j}.npz"
                    np.savez_compressed(
                        path_aug,
                        flow_x=x1a[...,0],
                        flow_y=x1a[...,1],
                        cell_prob=x1a[...,2],
                        roi_mask=x2a[...,0]
                    )
    
    with Pool(60) as pool:
        pool.map(process_single_roi, labels)

    print(
        len(os.listdir(f"{dir_dataset}/npz_{size_crop}/")), 
        len(os.listdir(f"{dir_dataset}/npz_{size_crop}_aug{j}/"))
    )