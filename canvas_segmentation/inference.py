import os
import collections
import numpy as np
import cv2
from torch.utils.data import DataLoader
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.callbacks import (InferCallback, CheckpointCallback)
from data import MMADataset
from albumentations import CenterCrop

class Infer:
  
    def __init__(self, image_folder, logs_dir, rez_dir, batch_size):
        super(Infer, self).__init__()
        self.image_folder = image_folder
        self.__logs_dir = logs_dir
        if os.path.exists(rez_dir) == False:
            os.mkdir(rez_dir)
            self.__rez_dir = rez_dir
        else:
            self.__rez_dir = rez_dir
        self.__batch_size = batch_size
        self.loader = MMADataset(self.image_folder,valid = True,
                                 transforms=False,test=True)

    def __get_data(self):
        loaders = collections.OrderedDict()
        loader = DataLoader(
            dataset=self.loader,
            batch_size=self.__batch_size,
            shuffle=False
        )
        loaders['infer'] = loader
        return loaders

    def inference(self,model):
        model = model
        loaders = self.__get_data()
        runner = SupervisedRunner()
        runner.infer(
            model=model,
            loaders=loaders,
            callbacks=[
                CheckpointCallback(
                    resume=os.path.join(self.__logs_dir, 'checkpoints/last.pth')
                ),
                InferCallback()
            ]
        )
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        for i, (input, output) in enumerate(zip(self.loader.image_list, runner.callbacks[1].predictions['logits'])):
            print(i)
            threshold = 0.6
            output = sigmoid(output)
            image_path = os.path.join(
                    self.image_folder,
                    "inputs",
                    input
                             )
            canvas_path = os.path.join(
                          self.image_folder,
                          "targets",
                          input
        )
            image = cv2.imread(image_path)
            target = cv2.imread(canvas_path)
            canvas = (output[0] > threshold).astype(np.uint8) * 255.
            #canvas = (output[0]* 255.).astype(np.uint8)
            original_height, original_width = image.shape[:2]
            canvas = CenterCrop(p=1,height=original_height, width=original_width)(image=canvas)["image"]
            cv2.imwrite(os.path.join(self.__rez_dir, f'predict{i}.jpg'), canvas)
            print(image_path)
            cv2.imwrite(os.path.join(self.__rez_dir, f'image{i}.jpg'), image)
            cv2.imwrite(os.path.join(self.__rez_dir, f'actual{i}.jpg'), target)