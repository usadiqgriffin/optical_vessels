#import lightning as L
from torch import nn
import torch
import os
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import logging
from pathlib import Path
from monai.losses.dice import DiceCELoss, DiceLoss, DiceFocalLoss, one_hot
from monai.losses.tversky import TverskyLoss
from utils import LossLogger
from tqdm import tqdm
#from tensorflow.keras.losses import BinaryCrossentropy

class ConvTranspose2dConsistent(nn.Module):
    def __init__(self, conv):
        super(ConvTranspose2dConsistent, self).__init__()
        self.conv = conv
        
    def forward(self, x):
        b, c, y, x = x.shape
        x = self.conv(x, output_size=(b, c//2, y*2, x*2))
        return x

class Models(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.debug = True

    def _forward(self, batch, criterion):
        '''same step in both training and validation, so we encapsulate it in this function'''
        
        loss_dict = {}
        X = batch['x'].to(self.device)
        T = batch['t'].to(self.device)  # not one-hot

        logging.debug(f"Model input shape:{X.shape}, target shape:{T.shape}")
        logits = self(X.float())  # one-hot, logits
        
        #print(f"Final output shape:{logits.shape}")
        softmax_layer = torch.nn.Softmax(dim=1)
        softmax = softmax_layer(logits)
        #del logits
        
        Y = torch.argmax(softmax, dim=1)
        T_one_hot = one_hot(T[:, None, ...], num_classes=self.output_classes)
        logging.debug(f"logits:{logits.shape}, softmax output:{softmax.shape}, T_one_hot:{T_one_hot.shape}")

        logging.debug(f"T:{T_one_hot.shape}")
        loss = self.criterion(softmax, T_one_hot)
        #loss, loss_dict = self.criterion(softmax, T_one_hot)
        logging.debug(f"Loss:{loss}")
        if batch["t"].isnan().any() or Y.isnan().any() or loss.isnan().any():
           logging.error("NaNs found in output or target or loss") 

        return Y, loss, softmax, loss_dict

    def setup(self, train_params):
        # Data definition

        # Dataloader:
            # resample to internal space
            # augmentation
            # 

        redux = "mean"

        # Loss and optimizer
        if train_params['loss'] == "DiceLoss":
            criterion = DiceLoss(include_background=False,
                                to_onehot_y=False, sigmoid=False, softmax=False, reduction=redux)
        elif train_params['loss'] == "DiceCELoss":
            criterion = DiceCELoss(include_background=False,
                                to_onehot_y=False, sigmoid=False, softmax=False, reduction=redux)
        elif train_params['loss'] == "CELoss":
                class_weights = torch.tensor([1, 100])
                criterion = nn.CrossEntropyLoss(weight=class_weights, reduction=redux)
        elif train_params['loss'] == "DiceFocalLoss":
            criterion = DiceFocalLoss(include_background=False,
                                to_onehot_y=False, sigmoid=False, softmax=False, reduction=redux)
        elif train_params['loss'] == "TverskyLoss":
            criterion = TverskyLoss(include_background=False,
                                to_onehot_y=False, sigmoid=False, softmax=False, reduction=redux)        
        else:
            raise ValueError

        optim = torch.optim.Adam(self.parameters(), lr=train_params['lr']) #, weight_decay=0.02)

        self.loss_logger = LossLogger()
        self.optim = optim

        self.criterion = criterion

    def train_model(self, train_dataloader, val_dataloader, train_params):

        # stuff common to most models
        # Main training loop
            # train and val step
            # save plots
            # save checkpoints
        
        latest_checkpoint_path = train_params['checkpoints_dir']
        latest_checkpoint_file = os.path.join(train_params['checkpoints_dir'], "model_latest.pt")
        device = train_params['device']
        save_every = train_params['save_every']

        self.device = device

        if not os.path.exists(latest_checkpoint_file):

            Path(latest_checkpoint_path).mkdir(parents=True, exist_ok=True)
            min_val_loss = float("Inf")
            logging.info(f"{train_params['checkpoints_dir']} not found, starting new experiment")
            # training loop init
            min_val_loss = float("inf")
            initial_epoch = 0
        else:  # resume experiment
            print(f"Previous checkpoint found at: {latest_checkpoint_file}")
            self.load_state_dict(torch.load(latest_checkpoint_file))
            self.loss_logger.load_from_csv(latest_checkpoint_path + "train_val.csv")
            val_loss_list = self.loss_logger.epoch_mean_losses_dict["val_loss"]
            latest_epoch = len(val_loss_list) * train_params['save_every']
            initial_epoch = latest_epoch + 1
            print(f"Starting from epoch: {initial_epoch}")
            min_val_loss = min(val_loss_list)
            print(f"Previous min val loss identified: {min_val_loss}")

        patches_per_patient = 1
        
        # Training loop
        for epoch in range(initial_epoch, train_params['max_epochs']):
            
            # train step
            for b_idx, batch in enumerate(tqdm(train_dataloader, f"Epoch {epoch}, training...")):
                
                super().train().to(device)
                self.optim.zero_grad()
                
                _, loss, _, _ = self._forward(batch, self.criterion)
                loss.backward()
                self.optim.step()

                self.loss_logger.record_batch_loss(
                    loss.detach().cpu().numpy(), "train_loss")
                
            logging.info(f"Epoch {epoch}, training loss:{loss.detach().cpu():0.4f}")
            if epoch % save_every == 0:
            # val step

                for b, batch in enumerate(tqdm(val_dataloader, f"Epoch {epoch}, validating...")):
                    with torch.no_grad():
                        pred, loss, pred_soft, loss_dict = self._forward(batch, self.criterion)
                    
                        self.loss_logger.record_batch_loss(
                            loss.detach().cpu().numpy(), "val_loss")
                        logging.info(f"Prediction sum:{np.sum(pred.detach().cpu().numpy())}")

                        # save loss components for plotting
                        for loss_comp in loss_dict.keys():
                            self.loss_logger.record_batch_loss(
                                loss_dict[loss_comp].detach().cpu().numpy(), loss_comp)

                        # save model i/o snapshots

                        '''for i in range(batch["index"].shape[0]):
                            index = int(batch["index"][i])
                            nvi_id = val_dataloader.dataset.nvi_ids[index]
                            preds = torch.stack((np.squeeze(pred_soft[i, 0, :, :, :], axis=1), pred[i, :, :, :]))
                            logging.debug(f"Preds size:{preds.shape}")
                            #if nvi_id in val_dataloader.dataset.nvi_ids_to_save_train_data and (val_dataloader.dataset.exported[index] == 0):
                            if nvi_id in val_dataloader.dataset.nvi_ids_to_save_train_data:

                                
                                self.io_snapshot_export(index, epoch, inputs = batch["inputs"][i, :, :, :, :], targets = batch["targets"][i, :, :, :], 
                                                        preds=preds, checkpoints_dir= experiment_train_params.checkpoints_dir)'''

                # reduce batches to epoch mean
                self.loss_logger.reduce_mean()
                mean_train_loss = self.loss_logger.epoch_mean_losses_dict["train_loss"][-1]
                mean_val_loss = self.loss_logger.epoch_mean_losses_dict["val_loss"][-1]
                mean_component_loss = {}
                for comp in loss_dict.keys():
                    mean_component_loss[comp] = self.loss_logger.epoch_mean_losses_dict[comp][-1]

                # save plots
                logging.info(f"Saving error plots in {train_params['checkpoints_dir']}")
                self.loss_logger.to_pandas().to_csv(train_params['checkpoints_dir'] + "train_val.csv")
                self.loss_logger.plot(os.path.join(train_params['checkpoints_dir']) + "train_val.png")

                # checkpoint
                torch.save(self.state_dict(), os.path.join(
                    train_params['checkpoints_dir'], f"model_latest.pt"))
                if (epoch % train_params['save_every']) == 0:
                    torch.save(self.state_dict(), os.path.join(
                        train_params['checkpoints_dir'], f"model_{epoch}.pt"))
                if mean_val_loss < min_val_loss:
                    best_checkpoint_path = os.path.join(
                        train_params['checkpoints_dir'], "model_best.pt")
                    print(
                        f"New min val loss found: {mean_val_loss:.3f}. Saving model to {best_checkpoint_path}")
                    torch.save(self.state_dict(), best_checkpoint_path)
                    min_val_loss = mean_val_loss

                print(
                    f"Epoch {epoch} | Train Loss: {mean_train_loss:.3f} | Val Loss: {mean_val_loss:.3f}")
                for comp in loss_dict.keys():
                    print(
                    f"{comp}: {mean_component_loss[comp]} |", end="")
                print()

class DownBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x

class UpBlock(nn.Module):

    def __init__(self, in_channels_up: int, in_channels_skip, out_channels: int):
        super().__init__()
        self.up_convtranspose = nn.Sequential(nn.ConvTranspose2d(in_channels_up, in_channels_up // 2,
                               kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.up_crb = nn.Sequential(
            nn.Conv2d(in_channels_up // 2 + in_channels_skip,
                      out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def _crop(self, x, target_shape: torch.Tensor.shape) -> torch.Tensor:
        x = x[:, :, :target_shape[-2], :target_shape[-1]]

        return x

    def forward(self, x_skip, x_up):

        x_up = self.up_convtranspose(x_up)
        x_up = self._crop(x_up, x_skip.shape)
        logging.debug(f"x_up:{x_up.shape}, x_skip:{x_skip.shape}")
        x_up = torch.concat([x_skip, x_up], dim=1)
        logging.debug(f"x concat:{x_up.shape}")
        x_up = self.up_crb(x_up)

        return x_up

class UNet2D(Models):
    
    def __init__(self, input_shape_with_channels, output_classes, depth, width):

        input_shape = input_shape_with_channels[1:] # x, y, z
        input_channels = input_shape_with_channels[0]

        # change to z, y, x for NP
        input_shape = input_shape[::-1]
        output_shape = [output_classes, input_shape[0], input_shape[1]] 
                                                                                                                
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        print('initializing model')
        self.internal_shape = input_shape
        self.input_channels = input_channels
        self.output_classes = output_classes
        layers = depth
        self.depth = layers
        self.width = width

        self.downblocks = nn.ModuleList()

        for i in range(self.depth):
            if i == 0:
                downblock_input_channels = input_channels
                downblock_output_channels = width * (2**(i + 2))
                stride = 1
            else:
                downblock_input_channels = width * (2**(i + 1))
                downblock_output_channels = width * (2**(i + 2))
                stride = 2
            if self.debug:
                print(
                    f"downblock_{i}, in: {downblock_input_channels}, out: {downblock_output_channels}, stride: {stride}")
            self.downblocks.append(DownBlock(
                in_channels=downblock_input_channels, out_channels=downblock_output_channels, stride=stride))

        self.upblocks = nn.ModuleList()
        for i in range(self.depth-1):
            in_channels_skip = width * (2**(i + 2))
            in_channels_up = width * (2**(i + 3))
            out_channels = width * (2**(i + 2))
            if self.debug:
                print(
                    f"upblock_{i}, in_channels_skip: {in_channels_skip}, in_channels_up: {in_channels_up}, out_channels: {out_channels}")
            self.upblocks.append(UpBlock(
                in_channels_skip=in_channels_skip,
                in_channels_up=in_channels_up,
                out_channels=out_channels,
            ))

        self.outconv = nn.Conv2d(in_channels=(
            width * (4)), out_channels=output_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x_skip = []
        for i in range(self.depth):
            logging.debug(f"Down {i}: {x.shape}-->")
            x = self.downblocks[i](x)
            x_skip.append(x)


        for i in range(self.depth-2, -1, -1):
            logging.debug(f"Up {i}: x:{x.shape}, skip:{x_skip[i].shape}-->")
            x = self.upblocks[i](x_skip=x_skip[i], x_up=x)

        x = self.outconv(x)

        '''for i in range(self.depth-1)[::-1]:
            logging.debug(f"Up {i}")
            logging.debug(f"Up {i}: x:{x.shape}, skip:{x_skip[i].shape}-->")
            x = self.upblocks[i](x_skip=x_skip[i], x_up=x)'''

        return x
    

