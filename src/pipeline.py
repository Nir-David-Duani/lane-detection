"""
Lane Detection Pipeline - Pre-processing and Line Detection
=============================================================
This module contains the first part of the pipeline:
1. ROI Masking
2. Color Thresholding
3. Edge Detection (Canny)
4. Line Detection (Hough Transform)

Everything after Hough is in line_detection.py
"""

import cv2
import numpy as np


# ============================================================
#                    1. ROI MASKING
# ============================================================

def apply_roi_mask(frame,
                   top_y_ratio=0.58,
                   left_bottom_ratio=0.05,
                   right_bottom_ratio=0.98,
                   top_left_x_ratio=0.25,
                   top_right_x_ratio=0.80):
    """
    Apply trapezoidal Region of Interest mask to focus on road area.
    
    Parameters tuned in: 02_roi_exploration.ipynb

    Args:
        frame: Input BGR image
        top_y_ratio: Vertical position of trapezoid top (0-1)
        left_bottom_ratio: Left edge position at bottom (0-1)
        right_bottom_ratio: Right edge position at bottom (0-1)
        top_left_x_ratio: Left edge position at top (0-1)
        top_right_x_ratio: Right edge position at top (0-1)
    
    Returns:
        masked: Image with ROI applied (black outside ROI)
        pts: Polygon points array for visualization
    """
    h, w = frame.shape[:2]
    
    # Define trapezoid vertices
    pts = np.array([[
        (int(w * left_bottom_ratio),  h),
        (int(w * right_bottom_ratio), h),
        (int(w * top_right_x_ratio),  int(h * top_y_ratio)),
        (int(w * top_left_x_ratio),   int(h * top_y_ratio)),
    ]], dtype=np.int32)
    
    # Create mask and apply
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, pts, (255, 255, 255))
    masked = cv2.bitwise_and(frame, mask)
    
    return masked, pts


# ============================================================
#                2. COLOR THRESHOLDING
# ============================================================

def apply_color_threshold(frame_roi):
    """
    Apply HSV color thresholding to detect white and yellow lane lines.
    
    Parameters tuned in: 03_color_thresholding.ipynb
    
    White lanes: High brightness (V), low saturation (S)
    Yellow lanes: Hue around 15-35 degrees
    
    Args:
        frame_roi: Input BGR image (typically after ROI masking)
    
    Returns:
        mask: Binary mask (255 = lane pixels, 0 = background)
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    
    # White lane detection: [H, S, V]
    # S <= 20 to avoid detecting white cars and gray areas
    lower_white = np.array([0, 0, 190])
    upper_white = np.array([180, 20, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Yellow lane detection: [H, S, V]
    # Adjusted to avoid detecting gray areas on the left side
    lower_yellow = np.array([15, 60, 50])  # Higher S and V to avoid gray detection
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine both masks
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask


# ============================================================
#                3. EDGE DETECTION (CANNY)
# ============================================================

def apply_canny(mask, low_threshold=50, high_threshold=150, blur_kernel=5):
    """
    Apply Canny edge detection to extract precise lane line edges.
    
    Parameters tuned in: 04_canny_edge_detection.ipynb
    
    Args:
        mask: Binary mask from color thresholding (grayscale, 0-255)
        low_threshold: Lower threshold for hysteresis (default: 50)
        high_threshold: Upper threshold for hysteresis (default: 150)
        blur_kernel: Gaussian blur kernel size, must be odd (default: 5)
    
    Returns:
        edges: Binary edge map (255 = edge, 0 = no edge)
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges


# ============================================================
#                4. LINE DETECTION (HOUGH TRANSFORM)
# ============================================================

def detect_lines_hough(edges, 
                       rho=1,
                       theta=np.pi/180,
                       threshold=30,
                       min_line_length=40,
                       max_line_gap=100):
    """
    Detect straight lines using Probabilistic Hough Transform.
    
    This function converts edge pixels into straight line segments.
    
    Parameters to tune in: 05_line_detection.ipynb
    
    Args:
        edges: Binary edge map from Canny detection
        rho: Distance resolution in pixels
        theta: Angle resolution in radians (np.pi/180 = 1 degree)
        threshold: Minimum number of intersections to detect a line
        min_line_length: Minimum length of line segment in pixels
        max_line_gap: Maximum gap between points to be considered same line
    
    Returns:
        lines: Array of line segments [[x1, y1, x2, y2], ...]
               Returns None if no lines detected
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    return lines