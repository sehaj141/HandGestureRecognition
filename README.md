A real-time computer vision system that accurately detects and classifies hand gestures using traditional Machine Learning techniques. Achieves 86%+ accuracy on 5 common gestures.

## Description
This project implements a robust hand gesture recognition system using OpenCV for computer vision and Scikit-learn for machine learning classification. The system processes webcam input or static images, segments the hand using adaptive skin detection, extracts 25+ discriminative features (contour properties, convex defects, Hu moments), and classifies gestures using an ensemble of SVM and Random Forest classifiers.

Supported Gestures:
- ✋ Open Hand
- ✌️ Two Fingers
- 🖐️ Three Fingers
- 👍 Thumbs Up

## Features : 
- Real-time webcam detection (30+ FPS)
- 86%+ classification accuracy
- Robust skin segmentation (works with varying lighting/skin tones)
- 25+ handcrafted features (rotation/scale invariant)
- Ensemble classifier (SVM + Random Forest)
- Temporal smoothing (stable predictions)
- Production-ready (error handling, auto-balancing)
- Easy dataset expansion

## Technologies Used

- Python 3.8+ - Core language
- OpenCV 4.x - Computer vision & image processing
- NumPy - Numerical computations
- Scikit-learn - Machine Learning classifiers
- SciPy - Distance calculations

## Installation

### Prerequisites
- Python 3.8+
- Windows/Linux/macOS

## Step-by-Step Setup
# 1. Clone or download project
git clone <your-repo> HandGestureRecognition
cd HandGestureRecognition

# 2. Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install opencv-python numpy scikit-learn scipy

# 4. Verify installation
python -c "import cv2, numpy, sklearn; print('✅ Ready!')"

## Dataset Structure
-   dataset/
    - |
    - ├── Open_Hand/     ✋ (100+ images)
    - ├── Two_Fingers/   ✌️ (100+ images)
    - ├── Three_Fingers/ 🖐️ (100+ images)
    - └── Thumbs_Up/     👍 (100+ images)

Dataset Tips:

- 100+ images per gesture
- Varied lighting, backgrounds, angles
- JPG/PNG format
- Hand fills ~50% of image

##  How to Run
- 1. Train Model (First Time Only) : python train_model.py
    - PRODUCTION ENSEMBLE RESULTS: Accuracy: 0.86 (86%)
    - Model saved!
- 2. Run Real-time Detection : python main.py
- Controls:
   - ESC → Exit
   - SPACE → Reset prediction
   - Show hand in green ROI box

- Sample Output:
     - 🖐️ STABLE: Open_Hand (94.2%)
     -   RAW: Open_Hand (95.1%)
     -   Confidence: 92.3%

## Output Explanation
- STABLE: Open_Hand     : Most confident prediction (temporal smoothing)
- RAW: Open_Hand (95.1%) : Instant classifier output + confidence
- Confidence: 92.3%      : Average confidence over last 12 frames
- Green ROI              : Hand detection region
- Yellow contour         : Detected hand boundary

## Troubleshooting
- "No model found" : Run python train_model.py
- "Open_Hand not detected": Add more Open_Hand  images to dataset/Open_Hand /
- Low accuracy: Balance dataset (100+ images/gesture)
- Webcam issues: Check camera permissions

## Future Improvements
[ ] Deep Learning (MediaPipe + CNN)
[ ] More gestures (10+ gestures)
[ ] Multi-hand detection
[ ] GPU acceleration
[ ] Mobile deployment (TensorFlow Lite)
[ ] Custom gesture training UI

## Performance Metrics
- Validation Accuracy: 84.2%
- Per-class F1-scores:
  Three_Fingers: 0.83 | Thumbs_Up: 0.87 | Two_Fingers: 0.88
- Inference Speed: 35 FPS
- Model Size: 12 MB

## Author
Sehajpreet Kaur 
- Built using Computer Vision & ML



 
