from os.path import join

from experiment import Experiment
from model import UNet2D
from dataloader import dataloader
from glob import glob
import random
from random import shuffle
import logging
import os


if __name__ == "__main__":
    # Initializing data

    
    data_dir = "data/diabetic-retinopathy-dataset/resized"
    dev_paths_list = glob(data_dir + "/train/*t.jpeg")[:500] # left or right, exclude masks
    test_paths_list = glob(data_dir + "/test/*t.jpeg")[:200]
    random.Random(4).shuffle(dev_paths_list)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"{len(dev_paths_list)} Training images found")

    train_paths_list, val_paths_list = dev_paths_list[:int(0.7*len(dev_paths_list))], dev_paths_list[int(0.7*len(dev_paths_list))+1:]
    
    model = UNet2D()
    model.define_model()
    model.initialize_weights(global_step=0)
    data = dataloader(train_paths_list, val_paths_list, test_paths_list)
    
    # Initializing experiment
    experiment = Experiment(model, data, "output")
    experiment.train()
