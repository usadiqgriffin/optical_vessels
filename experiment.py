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

class Experiment(object):

    def __init__(self, model, data, output_path):

        self.model = model
        self.output_path = output_path
        self.data = data

    def train(self):

        epochs = 200
        save_model_epoch = 20
        batch_size = 30
        internal_shape = self.model.internal_shape
        #internal_pixdim = self.model.internal_pixdim

        train_batches =  int(np.ceil(len(self.data.train_set) / batch_size))
        val_batches = int(np.ceil(len(self.data.val_set) / batch_size))

        print('\n**************************************')
        print('len(self.data.train_set) =', len(self.data.train_set))
        print('train_batches =', train_batches)
        print('len(self.data.val_set) =', len(self.data.val_set))
        print('val_batches =', val_batches)
        print('**************************************\n')

        # Possibly move these to experiment setup?
        train_options = {"max_epochs": 30,
            "checkpoints_dir": CHECKPOINT_DIR,
            "internal_shape": internal_shape,
            "save_every": 10,
            "device": 'cuda:0' if torch.cuda.is_available() else "cpu",
            "batch_size" : 3,
            "loss": "DiceCELoss",
            "lr": 1e-3
            }
        self.train_params = train_options

        train_dl = torch.utils.data.DataLoader(
            dataset=self.data,
            batch_size=train_options['batch_size'],
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        val_dl = torch.utils.data.DataLoader(
            dataset=self.data,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )

        self.model.setup(train_params=train_options)

        self.model.train_model(train_dataloader=train_dl, 
            val_dataloader=val_dl,
            train_params=train_options)

    def eval(self, load_mode):
        
        if load_mode == 1:
            # Forcing the model to run on the validation set only
            datasets = [[], self.data.val_set, []]
        else:
            # Running the model on all the train/val/test set, depending on what is loaded to memory (i.e. load_mode)
            datasets = [self.data.train_set, self.data.val_set, self.data.test_set]

        model_outputs = []  # will contain everything needed to run the test and generate the report
        loss0_arr = []

        for dataset in datasets:

            for i in range(len(dataset)):

                print(i, '/', len(dataset))

                # Validating the validation set
                image = np.expand_dims(dataset[i][0], 0)
                vessel_mask = np.expand_dims(dataset[i][1], 0)
                row = ''
                #pred_vessel_internal = dataset[i][3]


                trans_mat = np.expand_dims(np.zeros(shape=(3, 4)), 0)

                image_trans, vessel_mask_trans, pred_vessels, loss0 = self.model.sess.run(
                    [self.model.image_trans,
                     self.model.vessel_mask_trans,
                     self.model.pred_vessels,
                     self.model.loss0],
                    feed_dict={
                        self.model.image: image,
                        self.model.vessel_mask: vessel_mask,
                        self.model.trans_mat: trans_mat,
                        self.model.training: 0})

                model_outputs.append([image_trans,
                                      vessel_mask_trans,
                                      pred_vessels,
                                      row])
                loss0_arr.append(loss0)
        
        return model_outputs