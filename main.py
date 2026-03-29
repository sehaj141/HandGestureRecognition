import cv2
import numpy as np
from hand_detection import segment_hand
from feature_extraction import extract_features
from classifier import GestureClassifier
from collections import deque

# Initialize
cap = cv2.VideoCapture(0)
classifier = GestureClassifier()
classifier.load()

# Buffer for smoothing
gesture_buffer = deque(maxlen=10)

gesture_name = "None"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Define ROI
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)

    # Convert ROI to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)

        if contour is not None and len(contour) >= 5:

            features = extract_features(contour)

            if features is not None:
                prediction = classifier.predict(features)
                print(prediction)
                # Add to buffer
                gesture_buffer.append(prediction)

                # Get most frequent prediction
                gesture_name = max(set(gesture_buffer), key=gesture_buffer.count)

                # Draw contour (shifted to ROI position)
                contour = contour + np.array([100, 100])
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # Display gesture
    cv2.putText(frame, f"Gesture: {gesture_name}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()