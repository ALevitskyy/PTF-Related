from utils import make_video_from_frames,extract_frames
from inference import Infer
from datetime import datetime
import os
import segmentation_models_pytorch as smp
input_file = "ufc234_gastelum_bisping_1080p_nosound_cut.mp4"
output_file = "test.mp4"
intermediate_dir = "video_frames"
intermediate_dir2 = "video_frames_processed"
print("Frame extraction")
now = datetime.now()
extract_frames(input_file, intermediate_dir)
print(datetime.now()-now)
print("Inference")
inferer = Infer(rez_dir = intermediate_dir2 ,
        image_folder = intermediate_dir,
        batch_size = 3)
model = smp.Unet("se_resnext50_32x4d")
model.cuda()
inferer.inference(model)
print(datetime.now()-now)
print("Making video")
make_video_from_frames(
                frame_dir = os.path.join(intermediate_dir2, "mask"),
                target_path = output_file
        )
print(datetime.now()-now)
