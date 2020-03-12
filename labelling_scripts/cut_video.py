from scripts.cutting import get_image
from scripts.rand_timestamp import random_time_stamps
import subprocess


def add_directory(input_dir, directory_name):
    try:
        subprocess.check_output(["mkdir", input_dir + "/" + directory_name])
    except:
        pass


# INPUTS
video = "Top 20 Knockouts in UFC History-LWE79K2Ii-s.mp4"
timestamp_list = [
    ["00:00:23", "00:01:33"],
    ["00:02:02", "00:02:41"],
    ["00:03:05", "00:03:40"],
    ["00:04:30", "00:04:49"],
    ["00:05:21", "00:07:21"],
]
# INPUTS END
list_of_directiories = ["images", "mask_computer", "mask_human", "gimp_files"]
video2 = "videos/" + video
video_name = video.split(".")[0]
image_directory = "images/" + video_name
timestamps = random_time_stamps(timestamp_list)
for directory in list_of_directiories:
    add_directory(directory, video_name)
for timestamp in timestamps:
    filename = image_directory + "/" + timestamp + ".png"
    get_image(video2, timestamp, filename)
