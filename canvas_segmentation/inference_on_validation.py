from inference import Infer
from models.AlbuNet.AlbuNet import AlbuNet
import segmentation_models_pytorch as smp

inferer = Infer(threshold=0.75)
model = smp.Unet("se_resnext50_32x4d")
model.cuda()
inferer.inference(model)
