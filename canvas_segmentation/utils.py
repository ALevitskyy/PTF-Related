
import cv2
from typing import Tuple
import numpy as np
def make_overlay(
                img: np.ndarray, mask: np.ndarray,
                        color: Tuple[int, int, int] = (89, 69, 15),
                                alpha: float = 0.5
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
