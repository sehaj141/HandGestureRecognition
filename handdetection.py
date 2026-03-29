# handdetection.py
import cv2
import numpy as np

def segmenthand_adaptive(frame):
    """Adaptive hand segmentation (used in main.py)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Multiple skin tone ranges for robustness
    lower_skin1 = np.array([0, 20, 70])
    upper_skin1 = np.array([17, 255, 255])
    lower_skin2 = np.array([17, 70, 70])
    upper_skin2 = np.array([20, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    mask = mask1 | mask2
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask