import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

class LossLogger():

    def __init__(self) -> None:
        '''
            current_batch_losses_dict: dict of lists containing the batch losses by name
            epoch_mean_losses_dict: dict of lists containing the epoch losses by name
        '''
        self.current_batch_losses_dict = {}
        self.epoch_mean_losses_dict = {}
        self.deltax = {}

    def load_from_csv(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path, index_col="epochs")
        for c in df.columns:
            self.current_batch_losses_dict[c] = []
            self.epoch_mean_losses_dict[c] = df[c].to_list()

    def record_batch_loss(self, loss: float, name: str) -> None:
        if name not in self.current_batch_losses_dict.keys():
            self.current_batch_losses_dict[name] = []

        self.current_batch_losses_dict[name].append(loss)

    def reduce_mean(self) -> pd.DataFrame:
        # reduce batch losses to mean and append to dict
        for name, batch_losses_list in self.current_batch_losses_dict.items():
            if name not in self.epoch_mean_losses_dict.keys():
                self.epoch_mean_losses_dict[name] = []
            self.epoch_mean_losses_dict[name].append(np.array(batch_losses_list).mean())
            
            #self.epoch_mean_losses_dict[name].append(np.array(batch_losses_list))

        # flush the current batch losses
        self.current_batch_losses_dict = {}

    def to_pandas(self) -> pd.DataFrame:
        num_epochs = len(list(self.epoch_mean_losses_dict.values())[0])
        df = pd.DataFrame(data=self.epoch_mean_losses_dict,
                          index=[e for e in range(num_epochs)])
        df.index.name = "epochs"

        return df

    def plot(self, png_path: str) -> None:
        matplotlib.use("Agg")
        num_epochs = len(list(self.epoch_mean_losses_dict.values())[0])

        for name in self.epoch_mean_losses_dict.keys():
            y = self.epoch_mean_losses_dict[name]
            x = [e for e in range(len(y))]
            
            plt.plot(x, y)
        plt.legend(list(self.epoch_mean_losses_dict.keys()))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()
        plt.savefig(png_path)
        plt.close()