import cv2
import numpy as np

def segment_hand(frame):
    """Segments the hand region based on skin color in YCrCb color space."""
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    blur = cv2.GaussianBlur(ycrcb, (11, 11), 0)

    # Define skin color range
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(blur, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 1000:
        return None, None

    return roi, max_contour
