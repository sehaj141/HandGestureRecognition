# train_model.py
import os
import cv2
import numpy as np
from feature_extraction import extract_improved_features
from classifier import GestureClassifier
import random

DATASET_DIR = 'dataset'

def preprocess_training_image(img_path):
    """Enhanced preprocessing for training images"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Resize for consistency
    img = cv2.resize(img, (300, 300))
    
    # CLAHE for better contrast (helps segmentation)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return img

print("🤜🤛 Hand Gesture Training Dataset Collector")
print("=" * 50)

# Collect training data
X = []
y = []
gesture_samples = {}

for gesture in sorted(os.listdir(DATASET_DIR)):
    gesture_path = os.path.join(DATASET_DIR, gesture)
    if not os.path.isdir(gesture_path):
        continue
    
    print(f"\n📁 Processing '{gesture}'...")
    
    image_files = [f for f in os.listdir(gesture_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"   ❌ No images found in '{gesture}'")
        continue
    
    print(f"   📸 Found {len(image_files)} images")
    
    # Shuffle and limit for balance (max 150 per class)
    random.shuffle(image_files)
    image_files = image_files[:150]
    
    valid_count = 0
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(gesture_path, img_file)
        img = preprocess_training_image(img_path)
        
        if img is None:
            continue
        # IMPROVED Fist-friendly skin segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # BROADER skin ranges (works better for fists)
        skin_ranges = [
            ([0, 10, 60], [20, 255, 255]),    # Broad range 1
            ([0, 30, 60], [15, 255, 255]),    # Broad range 2
            ([15, 40, 70], [25, 255, 255])    # Darker skin
        ]
        
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in skin_ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # BETTER morphology for fists (less aggressive opening)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 1500:  # Good hand size
                features = extract_improved_features(contour)
                if features is not None:
                    X.append(features)
                    y.append(gesture)
                    valid_count += 1
        
        if i % 20 == 0:
            print(f"   ⏳ Processed {i}/{len(image_files)} images...")
    
    gesture_samples[gesture] = valid_count
    print(f"   ✅ Added {valid_count} valid samples for '{gesture}'")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print("\n" + "=" * 50)
print("📊 FINAL DATASET SUMMARY")
print("=" * 50)
print(f"Total valid samples: {len(X):,}")
print(f"Feature dimensions: {X.shape[1]}")
print(f"Gestures found: {len(np.unique(y))}")
print("\nPer-gesture breakdown:")
for gesture, count in sorted(gesture_samples.items()):
    print(f"  🤜 {gesture:12s}: {count:3d} samples")

if len(X) < 30:
    print("\n❌ ERROR: Insufficient data! Need at least 30 samples.")
    print("💡 Add more images to your dataset/ folders")
    exit(1)

print(f"\n🎯 Training ensemble classifier...")
print("This may take 30-60 seconds...")

# Train the model
classifier = GestureClassifier()
classifier.train(X, y)
classifier.save('gesture_classifier.pkl')

print("\n" + "=" * 50)
print("✅ TRAINING COMPLETE!")
print("💾 Model saved as 'gesture_classifier.pkl'")
print("\n🚀 Run 'python main.py' to test with webcam!")
print("=" * 50)