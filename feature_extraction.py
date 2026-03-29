# feature_extraction.py - NaN-PROOF VERSION
import cv2
import numpy as np
from scipy.spatial.distance import cdist

def extract_improved_features(contour):
    if contour is None or len(contour) < 5:
        return None
    
    contour = contour.astype(np.float32).reshape(-1, 1, 2)
    
    try:
        # Basic geometry - SAFE
        area = abs(cv2.contourArea(contour))
        if area < 100:
            return None
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None
        
        circularity = min(1.0, 4 * np.pi * area / (perimeter ** 2))
        
        # Convex hull - SAFE
        hull = cv2.convexHull(contour)
        hull_area = abs(cv2.contourArea(hull))
        convexity = min(1.0, hull_area / area) if area > 0 else 0.5
        
        # Defects - SAFE
        defect_count = 0
        defect_depth = 0.0
        try:
            hull_idx = cv2.convexHull(contour, returnPoints=False)
            if len(hull_idx) >= 3:
                defects = cv2.convexityDefects(contour, hull_idx)
                if defects is not None and len(defects) > 0:
                    defect_count = min(10, len(defects))  # Cap at 10
                    defect_depth = np.clip(np.mean(defects[:, 0, 3]) / 255.0, 0, 1)
        except:
            pass
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = np.clip(w / max(h, 1), 0.1, 10.0)
        extent = np.clip(area / (w * h), 0, 1) if w * h > 0 else 0.5
        
        # Hu moments - SAFE
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = np.clip(-np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7), -10, 10)
        
        # Ellipse - SAFE
        ellipse_ratio = 1.0
        try:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                ma, mi = ellipse[1]
                ellipse_ratio = np.clip(min(ma, mi) / max(ma, mi, 1e-6), 0.1, 1.0)
        except:
            pass
        
        # Projection profile
        y_min, y_max = cv2.boundingRect(contour)[1:3]
        height = max(1, y_max - y_min)
        proj = np.bincount(np.clip(contour[:, 0, 1].astype(int) - y_min, 0, height-1), 
                          minlength=height).astype(np.float32)
        
        proj_mean = np.mean(proj) / max(1, height)
        proj_std = np.std(proj) / max(1, height)
        proj_peaks = np.sum(proj > proj_mean * 1.5) / max(1, height)
        
        # Distances - SAFE
        max_dist = 0.0
        try:
            if len(contour) > 10:
                pts = contour[:, 0, :]
                dists = cdist(pts, pts)
                max_dist = np.max(dists) / max(np.sqrt(area), 1)
        except:
            pass
        
        compactness = np.clip(perimeter / max(np.sqrt(area * 4 * np.pi), 1), 0.1, 10)
        
        # FINAL SAFE FEATURES (all clamped, no NaN/inf)
        features = np.array([
            circularity, convexity, aspect_ratio, extent, defect_count/10.0,
            defect_depth, ellipse_ratio, compactness,
            proj_mean, proj_std, proj_peaks,
            *hu_moments[:7],  # First 7 Hu moments
            max_dist, perimeter/area, w/perimeter, h/perimeter
        ], dtype=np.float32)
        
        # FINAL VALIDATION - replace any remaining bad values
        features = np.nan_to_num(features, nan=0.5, posinf=1.0, neginf=0.0)
        features = np.clip(features, -5, 5)  # Safe range
        
        return features
        
    except:
        return None