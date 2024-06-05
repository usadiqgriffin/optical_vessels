from PIL import Image
import numpy as np
from skimage.filters.ridges import sato, frangi
from dataloader import OpticalDataloader
import random
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import logging
import matplotlib.pyplot as plt
from collections import namedtuple
from GLOBALS import *
import torch
from pathlib import Path

class Experiment(object):

    def __init__(self, model, output_path, dataset_list , modes_list):

        self.model = model
        self.output_path = output_path

        self.data = {}

        for i in range(len(dataset_list)):
            self.data[modes_list[i]] = dataset_list[i]

    def train(self):

        epochs = 300
        save_model_epoch = 20
        batch_size = 30
        internal_shape = self.model.internal_shape
        #internal_pixdim = self.model.internal_pixdim

        train_batches =  int(np.ceil(len(self.data['train']) / batch_size))
        val_batches = int(np.ceil(len(self.data['valid']) / batch_size))

        print('\n**************************************')
        print('len(self.data.train_set) =', len(self.data['train']))
        print('train_batches =', train_batches)
        print('len(self.data.val_set) =', len(self.data['valid']))
        print('val_batches =', val_batches)
        print('**************************************\n')

        # Possibly move these to experiment setup?
        train_options = {"max_epochs": 300,
            "checkpoints_dir": CHECKPOINT_DIR,
            "internal_shape": internal_shape,
            "save_every": 20,
            "device": 'cuda:0' if torch.cuda.is_available() else "cpu",
            "batch_size" : 30,
            "loss": "DiceCELoss",
            "lr": 1e-3
            }
        self.train_params = train_options
        Path(train_options["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)

        # Create training and validation dataloaders

        train_dl = torch.utils.data.DataLoader(
            dataset=self.data['train'],
            batch_size=train_options['batch_size'],
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        val_dl = torch.utils.data.DataLoader(
            dataset=self.data['valid'],
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )

        self.model.setup(train_params=train_options)
        self.model.train_model(train_dataloader=train_dl, 
            val_dataloader=val_dl,
            train_params=train_options)

    def eval(self):

        internal_shape = self.model.internal_shape
        deploy_options = {"checkpoints_dir": CHECKPOINT_DIR,
            "deploy_dir": EXPERIMENT_DEPLOY_DIR,
            "internal_shape": internal_shape,
            "save_every": 20,
            "device": 'cuda:0' if torch.cuda.is_available() else "cpu",
            "batch_size" : 1,
            "loss": "DiceCELoss",
            "lr": 1e-3
            }
        self.deploy_params = deploy_options
        Path(deploy_options["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
        Path(deploy_options["deploy_dir"]).mkdir(parents=True, exist_ok=True)

        # Setup model and dataset
        self.model.setup(train_params=deploy_options)
        test_dl = torch.utils.data.DataLoader(
            dataset=self.data['test'],
            batch_size=1,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
        model_outputs = self.model.deploy_model(exp_options=deploy_options, test_dataloader=test_dl)
        
        return model_outputs