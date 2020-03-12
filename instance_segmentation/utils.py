import ffmpy
import subprocess
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Union
import cv2
import numpy as np
from config import split_file
import os


def make_ground_truth(split_file=split_file, target_dir="ground_truth"):
    with open(split_file, "r") as f:
        image_list = eval(f.read())
    os.mkdir(target_dir)
    os.mkdir(os.path.join(target_dir, "mask"))
    os.mkdir(os.path.join(target_dir, "image"))
    print(image_list["valid"])
    for i in image_list["valid"]:
        file_path0 = i[0]
        file_path1 = i[1]
        image_name = file_path0.split("/")[-1]
        image = cv2.imread(file_path0)
        mask = cv2.imread(file_path1)
        cv2.imwrite(os.path.join(target_dir, "image", image_name), image)
        cv2.imwrite(os.path.join(target_dir, "mask", image_name), mask)


def make_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (89, 69, 15),
    alpha: float = 0.5,
) -> np.ndarray:
    # result img
    output = img.copy()
    # overlay mask
    overlay = np.zeros_like(img)
    overlay[:, :] = color
    # inverse mask
    mask_inv = cv2.bitwise_not(mask)
    # black-out the area of mask
    output = cv2.bitwise_and(output, output, mask=mask_inv)
    # take only region of mask from overlay mask
    overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
    # original img with opaque mask
    overlay = cv2.add(output, overlay)
    output = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    return output


def extract_frames(
    filename: Union[str, Path], output_dir: Union[str, Path], verbose: bool = True
) -> None:
    # create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # prepare source and target paths
    src_path = str(filename)
    target_path = str(output_dir / "%d.jpg")
    # input and output options
    input_opts = None
    # -b:v 10000k - average bitrate 10mb
    # -vsync 0 - all frames
    # -start_number 0 - start with frame number 0
    # -an - skip audio channels, use video only
    # -y - always overwrite
    # -q:v 2 - best quality for jpeg
    output_opts = "-start_number 0 -b:v 10000k -vsync 0 -an -y -q:v 2"
    # ffmpeg arguments
    inputs = {src_path: input_opts}
    outputs = {target_path: output_opts}
    # ffmpeg object
    ff = ffmpy.FFmpeg(inputs=inputs, outputs=outputs)
    # print cmd
    if verbose:
        print(f"ffmpeg cmd: {ff.cmd}")
    ff.run()


def make_video_from_frames(
    frame_dir: Union[str, Path],
    target_path: Union[str, Path],
    input_fps: Union[str, float] = "30000/1001",
    output_fps: Union[str, float] = "30000/1001",
    crf_quality: int = 17,
    img_ext: str = ".jpg",
    verbose: bool = True,
) -> None:
    # src path
    frame_dir = Path(frame_dir)
    src_path = str(frame_dir / f"%d{img_ext}")
    # target path - only .mp4
    target_path = Path(target_path).with_suffix(".mp4")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = str(target_path)
    # input and output options
    input_opts = f"-framerate {input_fps} -start_number 0"
    output_opts = [
        "-c:v",
        "libx264",
        "-vf",
        f"fps={output_fps}, format=yuv420p",
        "-crf",
        str(crf_quality),
        "-y",  # very important
    ]
    # ffmpeg arguments
    inputs = {src_path: input_opts}
    outputs = {target_path: output_opts}
    # ffmpeg object
    ff = ffmpy.FFmpeg(inputs=inputs, outputs=outputs)
    # print cmd
    if verbose:
        print(f"ffmpeg cmd: {ff.cmd}")
    ff.run()
