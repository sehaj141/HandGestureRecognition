import cv2
import os

def collect_gesture_images(gesture_name, save_dir='dataset', num_samples=200):
    """Capture images for dataset creation."""
    if not os.path.exists(f"{save_dir}/{gesture_name}"):
        os.makedirs(f"{save_dir}/{gesture_name}")

    cap = cv2.VideoCapture(0)
    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[100:400, 100:400]
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        cv2.putText(frame, f"Collecting {gesture_name}: {count}/{num_samples}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Dataset Collector", frame)

        img_path = f"{save_dir}/{gesture_name}/img_{count}.jpg"
        cv2.imwrite(img_path, roi)
        count += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
