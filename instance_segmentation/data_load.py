import subprocess
from config import dataset_url, split_file,\
        val_prop,image_folder
from glob import glob
import random
import json
import os
filename = dataset_url.split("/")[-1]
if os.path.exists(filename):
    #raise Exception("file already loaded")
    print("bla")
else:
    subprocess.call(["wget", dataset_url])
if os.path.exists(image_folder):
    #raise Exception("image folder  already exists")
    print("bla")
else:
    os.mkdir(image_folder)
    subprocess.call(["unzip", filename, "-d", image_folder])
#images = glob("./images/**/images/*.png",recursive=True)
masks = glob("./images/**/masks/*.png",recursive=True)
images = [path.replace("masks","images").strip(" ") for path in masks]
pairs = list(zip(images, masks))
random.shuffle(pairs)
split = int(len(pairs)*val_prop)
validation_list = pairs[:split]
train_list = pairs[split:]
file_content = {"valid":validation_list,
                "train":train_list}
with open(split_file,"w") as file:
    file.write(json.dumps(file_content))
