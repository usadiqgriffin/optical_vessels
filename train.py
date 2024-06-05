from os.path import join
import argparse

from experiment import Experiment
from model import UNet2D
from dataloader import OpticalDataloader
from glob import glob
import random
from random import shuffle
import logging
import os
from GLOBALS import *



if __name__ == "__main__":
    # Initializing data

    # Arguments and logger
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action="store_true", default=False)
    parser.add_argument('--step', default='train')

    logging.basicConfig(level=logging.INFO)
    
    # Data params
    internal_shape = [1, 512, 512] # C, W, H
    data_dir = "data/diabetic-retinopathy-dataset/resized"

    args = parser.parse_args()
    n_dev = 20 if args.dryrun else 500
    n_test = 10 if args.dryrun else 200
    n_train = int(n_dev * 0.7)
    dev_paths_list = glob(data_dir + "/train/*t.jpeg")[:n_dev]
    test_paths_list = glob(data_dir + "/test/*t.jpeg")[:n_test]
    random.Random(4).shuffle(dev_paths_list)
    random.Random(4).shuffle(test_paths_list)

    logging.info(f"{len(dev_paths_list)} Training images found")

    train_paths_list, val_paths_list = dev_paths_list[:n_train], dev_paths_list[n_train:]
    data = OpticalDataloader(train_paths_list, val_paths_list, test_paths_list)
    
    # Initializing experiment

    model = UNet2D(internal_shape, output_classes=2, depth=8, width=2)
    #model.define_model()
    #model.initialize_weights(global_step=0)

    if args.step == "train" or args.step == "a2z":
        experiment = Experiment(model, data, "output")
        experiment.train()

    elif args.step == "deploy" or args.step == "a2z":

        experiment = Experiment(model, data, "output")
        experiment.eval() 

        print("\n")
        logging.critical(f"Deployment finished, results saved in {experiment.deploy_params['deploy_dir']}")
