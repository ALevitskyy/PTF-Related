from inference import Infer
from datetime import datetime
import os
import segmentation_models_pytorch as smp
inferer1 = Infer(
                rez_dir = "inferred" ,
                image_folder = "inferred2/overlay",
                batch_size = 3,
                num_batches=2,
                batch_id = 0
                        )

inferer2 = Infer(
                rez_dir = "inferred" ,
                image_folder = "inferred2/overlay",
                batch_size = 3,
                num_batches=2,
                batch_id = 1
                        )
model = smp.Unet("se_resnext50_32x4d")
model.cuda()
inferer1.inference(model)
inferer2.inference(model)
