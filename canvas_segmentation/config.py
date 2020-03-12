dataset_url = "https://s3.amazonaws.com/al-deeplearn/2019_04_19.zip"
image_folder = "images"
val_prop = 0.2
split_file = "train_val_split.json"

sizes = {"XXS": [3, 3], "XS": [5, 5], "S": [10, 10], "M": [20, 20]}

template = {
    "data/augmentations/overlays/cage/1.png": [[13, 8], [1920, 1080]],
    "data/augmentations/overlays/cage/2.png": [[150, 60], [1920, 1080]],
}

image_dict = {
    "data/augmentations/overlays/info/1.png": [1080, 1920],
    "data/augmentations/overlays/info/2.png": [1080, 1920],
    "data/augmentations/overlays/info/3.png": [1080, 1920],
    "data/augmentations/overlays/info/4.png": [1080, 1920],
    "data/augmentations/overlays/info/5.png": [1080, 1920],
    "data/augmentations/overlays/info/6.png": [1080, 1920],
    "data/augmentations/overlays/info/7.png": [1080, 1920],
    "data/augmentations/overlays/info/8.png": [1080, 1920],
    "data/augmentations/overlays/info/9.png": [1080, 1920],
    "data/augmentations/overlays/info/10.png": [1080, 1920],
    "data/augmentations/overlays/info/11.png": [1080, 1920],
    "data/augmentations/overlays/info/12.png": [1080, 1920],
}
