"""
Lane Detection Pipeline
========================
This module contains all the core functions for the lane detection pipeline.
Each function is tuned and validated in the corresponding Jupyter notebook.

Pipeline stages:
1. ROI Masking - apply_roi_mask()
2. Color Thresholding - apply_color_threshold()
3. Edge Detection - apply_canny()
4. Line Detection - (coming soon)
5. Lane Fitting - (coming soon)
"""

import cv2
import numpy as np


# ============================================================
#                    1. ROI MASKING
# ============================================================

def apply_roi_mask(frame,
                   top_y_ratio=0.60,
                   left_bottom_ratio=0.05,
                   right_bottom_ratio=0.95,
                   top_left_x_ratio=0.40,
                   top_right_x_ratio=0.60):
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
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Yellow lane detection: [H, S, V]
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine both masks
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Fill gaps
    
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
#                  4. VISUALIZATION HELPERS
# ============================================================

def draw_roi_outline(frame, pts, color=(0, 255, 0), thickness=3):
    """
    Draw ROI polygon outline on frame for visualization.
    
    Args:
        frame: BGR image to draw on (will be modified)
        pts: Polygon points from apply_roi_mask()
        color: BGR color tuple
        thickness: Line thickness
    
    Returns:
        frame: Frame with ROI outline drawn
    """
    cv2.polylines(frame, pts, isClosed=True, color=color, thickness=thickness)
    return frame


def overlay_mask_on_frame(frame, mask, color=(0, 255, 0), alpha=0.3):
    """
    Overlay binary mask on original frame with transparency.
    
    Args:
        frame: Original BGR image
        mask: Binary mask (grayscale)
        color: BGR color for overlay
        alpha: Transparency (0=invisible, 1=opaque)
    
    Returns:
        result: Frame with colored overlay
    """
    result = frame.copy()
    overlay = np.zeros_like(frame)
    overlay[mask == 255] = color
    result = cv2.addWeighted(result, 1, overlay, alpha, 0)
    return result
