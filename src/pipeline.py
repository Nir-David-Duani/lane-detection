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

def apply_roi_mask(
    frame,
    top_y_ratio,
    left_bottom_ratio,
    right_bottom_ratio,
    top_left_x_ratio,
    top_right_x_ratio,
):
    """
    Apply a trapezoidal Region of Interest mask.
    
    This is a generic geometric operation that uses normalized
    ratios (0–1) along image width/height. It does not assume
    any specific camera or road layout – those are encoded only
    in the ratios passed in by the caller.
    
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
    
    # Define trapezoid vertices (purely from ratios)
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

def apply_color_threshold(
    frame_roi,
    white_lower,
    white_upper,
    yellow_lower,
    yellow_upper,
    morph_kernel_size,
    morph_open_iter,
    morph_close_iter,
    morph_dilate_iter,
):
    """
    Generic HSV color thresholding + morphology.
    
    All numeric parameters (HSV bounds, kernel size, iterations) are
    provided by the caller so this function is independent of any
    specific task (lane detection, object detection, etc.).
    
    Args:
        frame_roi: Input BGR image (typically after ROI masking)
        white_lower: Lower HSV bound (tuple-like of 3 ints)
        white_upper: Upper HSV bound (tuple-like of 3 ints)
        yellow_lower: Lower HSV bound (tuple-like of 3 ints)
        yellow_upper: Upper HSV bound (tuple-like of 3 ints)
        morph_kernel_size: Size of square kernel for morphology ops
        morph_open_iter: Iterations for opening
        morph_close_iter: Iterations for closing
        morph_dilate_iter: Iterations for dilation
    
    Returns:
        mask: Binary mask (255 = foreground pixels, 0 = background)
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    
    # Convert bounds to numpy arrays
    lower_white = np.array(white_lower, dtype=np.uint8)
    upper_white = np.array(white_upper, dtype=np.uint8)
    lower_yellow = np.array(yellow_lower, dtype=np.uint8)
    upper_yellow = np.array(yellow_upper, dtype=np.uint8)
    
    # Threshold for both ranges
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine both masks
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Morphological operations
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    if morph_open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iter)
    if morph_close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)
    if morph_dilate_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=morph_dilate_iter)
    
    return mask


# ============================================================
#                3. EDGE DETECTION (CANNY)
# ============================================================

def apply_canny(mask, low_threshold, high_threshold, blur_kernel):
    """
    Generic Canny edge detection with pre-blur.
    
    All numeric thresholds and kernel sizes are controlled by the caller.
    
    Args:
        mask: Binary or grayscale image (0-255)
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis
        blur_kernel: Gaussian blur kernel size (must be odd)
    
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

def detect_lines_hough(
    edges,
    rho,
    theta,
    threshold,
    min_line_length,
    max_line_gap,
):
    """
    Generic straight-line detection using Probabilistic Hough Transform.
    
    All Hough parameters are passed in so this function is independent
    of any particular application.
    
    Args:
        edges: Binary edge map from Canny detection
        rho: Distance resolution in pixels
        theta: Angle resolution in radians
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
        maxLineGap=max_line_gap,
    )
    
    return lines