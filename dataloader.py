from PIL import Image
import numpy as np
from skimage.filters.ridges import sato, frangi
from tqdm import tqdm
import os
import logging
import torch

def zscore_norm(image):
    mean_val = np.mean(image)
    std_val = np.std(image)
    norm_image = (image - mean_val) / std_val

    return norm_image

def vessel_mask_ridge(img_np, filter_name = "sato"):

    sig = [2]
    
    if filter_name == "sato":
        img_ret = (img_np > 2) + (img_np <=-1)
        ridge = sato(img_np, sigmas=sig)
        mask = (1 - img_ret) * (ridge>0.05)

    else:
        #ridge = frangi(img_np, sigmas=[12, 14, 16, 18])
        sig = [2, 4, 6]
        ridge = frangi(img_np, sigmas=sig)
        img_ret = (img_np > 150) + (img_np <=0)
        mask = (1 - img_ret) * (ridge>0.03)

    return mask

class OpticalDataloader(torch.utils.data.Dataset):
    def __init__(self, train_paths_list, val_paths_list, test_paths_list=[]):

        # Initializing the datasets
        self.data = {}
        self.mode = ""

        self.load_data(train_paths_list, val_paths_list, test_paths_list)

    def load_data(self, train_paths_list, val_paths_list, test_paths_list=[]):
        
        internal_shape = [512, 512]
        datasets = [train_paths_list, val_paths_list, test_paths_list]
        recalc = False
        # Loading the appropriate data to memory
        # counter = 0
        train_set = []
        val_set = []
        test_set = []


        for i in range(len(datasets)):
            for j in range(len(datasets[i])):

                image_path = datasets[i][j]
                mask_path = image_path.replace(".jpeg", "_mask.jpeg")
                image_np = np.squeeze(np.array(Image.open(image_path).convert('L')))
                image_np = zscore_norm(image_np)
                
                if os.path.isfile(mask_path) and not recalc:
                    #vessel_mask_np = np.array(Image.open(mask_path).convert('L')).astype(np.uint8)
                    vessel_mask_np = np.array(Image.open(mask_path).convert('1'))
                else:
                    vessel_mask_np = vessel_mask_ridge(image_np, filter_name="sato")
                    vessel_mask = Image.fromarray((vessel_mask_np * 255).astype(np.uint8))
                    logging.debug(f"Writing vessel mask to {mask_path}")
                    vessel_mask.save(mask_path)

                #assert image_np.shape == internal_shape
                #assert vessel_mask_np.shape == internal_shape
                example = [image_np, vessel_mask_np]

                if i == 0:
                    train_set.append(example)
                elif i == 1:
                    val_set.append(example)
                elif i == 2:
                    test_set.append(example)

        self.data['train'] = train_set
        self.data['val'] = val_set
        self.data['test'] = test_set

        print()
        print('**************************************')
        print('len(train_set) =', len(train_set))
        print(f"image shape ={image_np.shape}, mask shape:{vessel_mask_np.shape}")
        print('len(val_set) =', len(val_set))
        print('len(test_set) =', len(test_set))
        print('Done Loading')
        print('**************************************')
    
    def __getitem__(self, index):


        x, t = self.data[self.mode][index]

        item = {}
        item["x"]= torch.unsqueeze(torch.Tensor(x), 0)
        item["t"]= torch.Tensor(t)
        return item
        
    def __len__(self):
        return len(self.data[self.mode])



