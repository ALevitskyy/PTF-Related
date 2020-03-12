import os
import shutil
from copy import deepcopy

maskdir = "data"
input_ind = 0
for subdir, dirs, files in os.walk(maskdir):
    for file in files:
        target_path = os.path.join(subdir, file)
        split = target_path.split(os.path.sep)
        targets = deepcopy(split[2])
        split[2] = "inputs"
        input_path = os.path.join(*split)
        print(targets)
        if targets == "targets":
            shutil.copy(
                target_path,
                os.path.join("dataset", targets, "frame" + str(input_ind) + ".png"),
            )
            shutil.copy(
                input_path,
                os.path.join("dataset", "inputs", "frame" + str(input_ind) + ".png"),
            )

            input_ind += 1
