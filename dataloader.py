from PIL import Image
import numpy as np
from skimage.filters.ridges import sato, frangi
from tqdm import tqdm
import os
import logging
import torch
from torchvision import transforms as T
from torchvision.transforms import v2

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
    def __init__(self, paths_list, mode = "train"):

        # Initializing the datasets
        self.data_list = []
        self.mode = mode

        self.load_data(paths_list)

    def load_data(self, paths_list):
        
        self.internal_shape = [512, 512]
        datasets = [paths_list]
        recalc = False
        # Loading the appropriate data to memory
        # counter = 0
        train_set = []

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

                train_set.append(example)

        self.data_list = train_set

        print()
        print('**************************************')
        print('len(train_set) =', len(train_set))
        print('Done Loading')
        print('**************************************')
    

    def augment_2D(self, image, flip_x, flip_y, angle, interp): 
        
        angle_xy = angle[0]
        image = image.cuda()
        
        # Random horizontal flipping
        if self.flip_y and flip_y > 0.5:
            image = TF.hflip(image)

        # Random vertical flipping
        if self.flip_x and flip_x > 0.5:
            image = TF.vflip(image)
        
        image = TF.rotate(image, angle_xy, interpolation=interp)

        return image

    def tv_augment_3D(self, image, mask):

        C, D, W, H = image.shape
    
        # Random affine
        max_angle_xy = 30
        angle_xy = random.random() * max_angle_xy - max_angle_xy / 2
        
        flip_x = random.random()
        flip_y = random.random()

        image = self.augment_2D(image.squeeze(), flip_x=flip_x, flip_y=flip_y, angle=(angle_xy), interp=TF.InterpolationMode.BILINEAR)
        mask =  self.augment_2D(mask.squeeze(), flip_x=flip_x, flip_y=flip_y, angle=(angle_xy), interp=TF.InterpolationMode.NEAREST)

        aug_dict = {}
        aug_dict['flip_x']=flip_x
        aug_dict['flip_y']=flip_y
        aug_dict['angle_xy']=angle_xy
        aug_dict['angle_yz']=angle_yz

        return image, mask, aug_dict

    def augment(self, image, mask):

        logging.debug(f"Augmenting ..")

        mask = torch.unsqueeze(mask, 0)
        # Define the transformation
        '''transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomVerticalFlip(p=0.5),
            T.GaussianBlur(kernel_size=3),  # You can adjust kernel size as needed
            T.RandomResizedCrop(size=(224, 224), scale=(0.99, 1.0), ratio=(0.75, 1.333)),
            T.ToTensor(),
            T.RandomErasing(p=0.2, scale=(0.05, 0.05), ratio=(0.5, 0.5)),
        ])'''
        
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=30),
            v2.RandomErasing(p=0.2, ratio=(0.5, 0.5)),
            v2.RandomResizedCrop(size=(512, 512), ratio=(0.75, 1.333)),
            #v2.RandomAffine(degrees=0, translate=[0.1, 0.1], shear=0)
            ]
        )
        
        images = torch.vstack([image, mask])
        images = transforms(images)

        augmented_image, augmented_mask = images[0, :, :], images[1, :, :]
        #augmented_mask = transforms(mask)

        return augmented_image, torch.squeeze(augmented_mask)

    def __getitem__(self, index):

        x, t = self.data_list[index]
        item = {}
        x = torch.unsqueeze(torch.Tensor(x), 0)
        t = torch.Tensor(t)

        logging.debug(f"\n \nDATA MODE: {self.mode}")

        if self.mode == "train":
            x, t = self.augment(x, t)
        
        item['x'] = x
        item['t'] = t
        
        return item
        
    def __len__(self):
        return len(self.data_list)



