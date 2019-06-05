from data.MMADataset import MMADataset
from data.InferenceDataset import InferenceDataset
from data.final_transforms import transforms, test_transform
from config import split_file
def dataset_test():
    dataset1 = MMADataset(split_file, None, True)
    dataset2 = MMADataset(split_file, transforms, False)
    print("validation set")
    print(dataset1.__len__())
    item = dataset1.__getitem__(1)
    valid_image = item["features"]
    valid_mask = item["targets"]
    print(valid_image.shape)
    print(valid_mask.shape)
    print("train set")
    print(dataset2.__len__())
    item = dataset2.__getitem__(1)
    valid_image = item["features"]
    valid_mask = item["targets"]
    print(valid_image.shape)
    print(valid_mask.shape)
#dataset_test()
def inference_dataset_test():
    folder = "images/2019_04_19/1_good/images"
    dataset1 = InferenceDataset(image_folder = folder,
                    transforms = test_transform)
    dataset2  = MMADataset(split_file = split_file,
                            transforms = test_transform,
                            valid = True)
    print("folder set")
    print(dataset1.__len__())
    item = dataset1.__getitem__(1)
    valid_image = item["features"]
    print(valid_image.shape)
    print("validation set")
    print(dataset2.__len__())
    item = dataset2.__getitem__(1)
    valid_image = item["features"]
    print(valid_image.shape)
inference_dataset_test()
