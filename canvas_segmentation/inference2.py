import os
import collections
import numpy as np
import cv2
from torch.utils.data import DataLoader
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.callbacks import (InferCallback, CheckpointCallback)
from data.MMADataset import MMADataset
from data.InferenceDataset import InferenceDataset
from albumentations import CenterCrop
from data.final_transforms import test_transform
from config import split_file
from utils import make_overlay
from models.AlbuNet.AlbuNet import AlbuNet

class Infer:

    def __init__(self,
                image_folder = None,
                logs_dir = "log",
                rez_dir = "inferred",
                batch_size = 1,
                threshold = 0.5,
                num_batches=1,
                batch_id=0,
                num_workers = 3):
        super(Infer, self).__init__()
        #If folder is not specified work with validation pics
        if image_folder is not None:
            self.loader = InferenceDataset(
                    image_folder = image_folder,
                    transforms = test_transform,
                    num_batches=num_batches,
                    batch_id = batch_id)
        else:
            self.loader = MMADataset(
                                  split_file = split_file,
                                  transforms = test_transform,
                                                  valid = True)
        self.__logs_dir = logs_dir
        if not os.path.exists(rez_dir):
            os.mkdir(rez_dir)
            self.__rez_dir = rez_dir
        else:
            self.__rez_dir = rez_dir
        self.__batch_size = batch_size
        self.threshold = threshold
        self.num_workers = num_workers

    def __get_data(self):
        loaders = collections.OrderedDict()
        loader = DataLoader(
            dataset=self.loader,
            batch_size=self.__batch_size,
            shuffle=False,
            num_workers= self.num_workers
        )
        loaders['infer'] = loader
        return loaders

    def inference(self,model):
        if not os.path.exists((os.path.join(self.__rez_dir, "mask"))):
            os.mkdir(os.path.join(self.__rez_dir, "mask"))
        if not os.path.exists((os.path.join(self.__rez_dir, "overlay"))):
            os.mkdir(os.path.join(self.__rez_dir,"overlay"))
        model = model
        loaders = self.__get_data()
        runner = SupervisedRunner()
        runner.infer(
            model=model,
            loaders=loaders,
            verbose = True,
            callbacks=[
                CheckpointCallback(
                    resume=os.path.join(self.__logs_dir, 'checkpoints/last.pth')
                ),
                InferCallback()
            ]
        )
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        for i, (input, output) in enumerate(zip(self.loader.image_list, runner.callbacks[1].predictions['logits'])):
            threshold = self.threshold
            output = sigmoid(output)
            image_path = input
            file_name = image_path[0].split("/")[-1]
            image = cv2.imread(image_path[0])
            canvas = (output[0] > threshold).astype(np.uint8) * 255
            canvas = np.squeeze(canvas)
            original_height, original_width = image.shape[:2]
            canvas = CenterCrop(p=1,height=original_height, width=original_width)(image=canvas)["image"]
            canvas = np.reshape(canvas,list(canvas.shape)+[1])
            overlay = make_overlay(image, canvas)
            cv2.imwrite(os.path.join(self.__rez_dir, "mask",file_name), canvas)
            cv2.imwrite(os.path.join(self.__rez_dir, "overlay", file_name), overlay)
