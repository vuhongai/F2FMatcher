import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from configs import *
from FM_dataset import *
from FM_models import *
from FM_train import *
set_seed(random_seed)


images_val = ["TA2c1", "TA21c3", "TA28c1"]

def define_set(f_name, images_val):
    count=0
    for i in images_val:
        if i in f_name:
            count+=1
    if count==0:
        return "train"
    else:
        return "val"

dir_ori = f"{dir_dataset}/npz_256"
dir_aug = f"{dir_dataset}/npz_256_aug0"

files_ori = os.listdir(dir_ori)
files_aug = os.listdir(dir_aug)

list_path_crop_ori_train = [] 
list_path_crop_aug_train = [] 
list_path_crop_ori_val = [] 
list_path_crop_aug_val = [] 

for f in tqdm(files_ori):
    f_base = f.split(".npz")[0]
    split = define_set(f_base, images_val)
    f_aug = f"{f_base}_aug0.npz"
    
    path_ori = f'{dir_ori}/{f_base}.npz'
    if f_aug in files_aug:
        path_aug = f'{dir_aug}/{f_aug}'
    else:
        path_aug = path_ori

    globals()[f'list_path_crop_ori_{split}'].append(path_ori)
    globals()[f'list_path_crop_aug_{split}'].append(path_aug)


f_test = "22-082_10X_WGA488_MD1-594_3-Scene-07-TA28c1-Image Export-48_s0c1x0-7536y0-7535_roi_2300"
list_path_crop_ori_test = [f'{dir_ori}/{f_test}.npz']
list_path_crop_aug_test = [f'{dir_aug}/{f_test}_aug0.npz']


if __name__ == '__main__':
    TrainFMmodel(
        list_path_crop_ori_train, 
        list_path_crop_aug_train,
        list_path_crop_ori_val, 
        list_path_crop_aug_val,
        list_path_crop_ori_test,
        list_path_crop_aug_test,
        batch_size = 256,
        n_log_train = 5,
        n_epoch = 30,
        learning_rate = 1e-5,
        checkpoint_path = f"{checkpoint_dir}/LatentVAE1_{size_crop}_{size_crop_resize}.pth",
        log_path="./logs/training_log_2nd.txt",
        re_train=True, 
        checkpoint_path_previous = f"{checkpoint_dir}/LatentVAE2_{size_crop}_{size_crop_resize}.pth",
    )




