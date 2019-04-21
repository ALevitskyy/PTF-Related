import os
import collections
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from catalyst.contrib.criterion import FocalLoss
from catalyst.dl.experiments.runner import SupervisedRunner
from catalyst.dl.callbacks import (
    LossCallback, TensorboardLogger, OptimizerCallback, CheckpointCallback,  ConsoleLogger)

from data.MMADataset import MMADataset
from config import split_file
from data.final_transforms import transforms, valid_transforms
from models.AlbuNet.AlbuNet import AlbuNet
class Model:
    def __init__(self, logs_dir="log", batch_size=32, workers=3):
        self.data = split_file
        if os.path.exists(logs_dir) == False:
            os.mkdir(logs_dir)
        self.logs_dir = logs_dir
        self.batch_size = batch_size
        self.workers = workers
        self.classes = ['canvas']

    def get_data(self):
        loaders = collections.OrderedDict()
        train_loader = DataLoader(
            dataset=MMADataset(self.data,
                transforms = transforms,
                valid=False),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers
        )
        valid_loader = DataLoader(
            dataset=MMADataset(self.data,
                       transforms = valid_transforms,
                       valid=True),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers
        )

        loaders['train'] = train_loader
        loaders['valid'] = valid_loader

        return loaders

    def set_callbacks(self):
        callbacks = [
                LossCallback(),
                CheckpointCallback(save_n_best=2),
                OptimizerCallback(),
                ConsoleLogger(),
                TensorboardLogger()
                ]
        return callbacks

    def train(self,model, epoch):
        model = model
        criterion = FocalLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        callbacks = self.set_callbacks()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)
        loaders = self.get_data()

        runner = SupervisedRunner()

        runner.train(
            model = model,
            criterion = criterion,
            loaders = loaders,
            logdir = self.logs_dir,
            optimizer = optimizer,
            scheduler = scheduler,
            num_epochs = epoch,
            verbose = True,
            callbacks = callbacks
        )

net = AlbuNet()
net.cuda()
model = Model()
model.train(net, 30)
