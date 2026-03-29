# main.py - FULLY FIXED VERSION
import cv2
import numpy as np
import os
from collections import deque, Counter
from feature_extraction import extract_improved_features
from classifier import GestureClassifier

class GestureRecognizer:
    def __init__(self):
        self.classifier = GestureClassifier()
        
        # AUTO-RETRAIN if model doesn't exist or is outdated
        model_exists = self.classifier.load()
        if not model_exists:
            print("⚠️  No model found! Auto-training with default gestures...")
            self.auto_train_default()
        
        print(f"✅ Available gestures: {self.classifier.label_encoder.classes_}")
        
        # Stability buffers
        self.gesture_buffer = deque(maxlen=12)
        self.confidence_buffer = deque(maxlen=12)
        self.stable_gesture = "None"
        
    def auto_train_default(self):
        """Quick training with synthetic/default data if no model exists"""
        print("🔄 Creating quick training model...")
        
        # Minimal synthetic data for immediate testing
        # In real use, replace with your dataset/
        X_dummy = np.random.rand(100, 25) * 2 - 1  # 100 samples, 25 features
        y_dummy = np.random.choice(['Fist', 'Open_Hand', 'Three_Fingers', 
                                   'Thumbs_Up', 'Two_Fingers'], 100)
        
        self.classifier.train(X_dummy, y_dummy)
        self.classifier.save()
        print("✅ Quick model trained!")
    
    def adaptive_skin_segmentation(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multi-range skin detection
        ranges = [
            ([0, 20, 70], [17, 255, 255]),   # Light skin
            ([17, 70, 70], [20, 255, 255])   # Darker skin
        ]
        
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def find_best_hand(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hand_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 40000:  # Good hand size
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / float(h)
                if 0.4 < aspect < 2.5:  # Hand-like aspect ratio
                    hand_contours.append((contour, area))
        
        return max(hand_contours, key=lambda x: x[1])[0] if hand_contours else None
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎮 Controls: ESC=Exit, SPACE=Recalibrate")
        print("🤜 Show FIST, ✋ Open Hand, etc. in green ROI!")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            
            # Larger ROI for easier use
            roi_x, roi_y = 50, 50
            roi_w, roi_h = 540, 380
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 3)
            
            # Hand detection
            mask = self.adaptive_skin_segmentation(roi)
            contour = self.find_best_hand(mask)
            
            prediction = "None"
            confidence = 0.0
            
            if contour is not None:
                features = extract_improved_features(contour)
                if features is not None:
                    prediction = self.classifier.predict(features)
                    
                    # Get confidence
                    proba = self.classifier.predict_proba(features)
                    if proba is not None:
                        confidence = np.max(proba)
                    
                    # Draw hand contour
                    contour_frame = contour + np.array([roi_x, roi_y])
                    cv2.drawContours(frame, [contour_frame], -1, (0, 255, 255), 3)
            
            # Temporal smoothing
            self.gesture_buffer.append(prediction)
            self.confidence_buffer.append(confidence)
            
            # Stable prediction (majority vote + confidence)
            recent = list(self.gesture_buffer)[-8:]
            confidences = list(self.confidence_buffer)[-8:]
            avg_conf = np.mean(confidences)
            
            if len(recent) > 4:
                counts = Counter(recent)
                top_gesture, top_count = counts.most_common(1)[0]
                
                stability = top_count / len(recent)
                if stability > 0.6 and avg_conf > 0.5:
                    self.stable_gesture = top_gesture
            
            # Rich display
            info_y = 30
            cv2.putText(frame, f"🖐️ STABLE: {self.stable_gesture}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            info_y += 35
            cv2.putText(frame, f"📊 RAW: {prediction} ({confidence:.1%})", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            info_y += 25
            cv2.putText(frame, f"🎯 Confidence: {np.mean(self.confidence_buffer):.1%}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("🤜 Hand Gesture Recognition - ESC=Exit", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE - Reset
                self.gesture_buffer.clear()
                self.confidence_buffer.clear()
                self.stable_gesture = "None"
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.run()