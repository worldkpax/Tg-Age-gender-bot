import numpy as np
import cv2

def bytes_to_cv2(b: bytes) -> np.ndarray:
    arr = np.asarray(bytearray(b), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img