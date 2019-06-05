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
num_batches = 70
model = smp.Unet("se_resnext50_32x4d")
model.cuda()
for i in range(0,num_batches):
    inferer1 = Infer(
        rez_dir = intermediate_dir2,
        image_folder = intermediate_dir,
        batch_size = 2,
        num_batches = num_batches,
        batch_id = i,
        threshold = 0.5)
    inferer1.inference(model)
print(datetime.now()-now)
print("Making video")
make_video_from_frames(
                frame_dir = os.path.join(intermediate_dir2, "mask"),
                target_path = output_file
        )
print(datetime.now()-now)
