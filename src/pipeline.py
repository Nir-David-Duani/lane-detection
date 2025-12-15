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
    # Balanced thresholds to detect white lanes while avoiding gray road surface:
    # - V ≥ 190 (sweet spot between too strict and too loose)
    # - S ≤ 30 - pure whites, minimal gray colors
    lower_white = np.array([0, 0, 190])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Yellow lane detection: [H, S, V]
    # Balanced thresholds for better yellow detection:
    # - S ≥ 70 (lowered from 100) - detect less saturated/faded yellows
    # - V ≥ 60 (lowered from 70) - detect darker/distant yellow lines
    lower_yellow = np.array([15, 70, 60])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine both masks
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Morphological operations - strengthened to fill gaps in lane lines
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fill gaps (increased from 1 to 2)
    
    # Additional dilation to strengthen and connect lane segments
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
    
    This function converts edge pixels into straight line segments,
    making the detected lanes appear much straighter.
    
    Parameters to tune in: 05_line_detection.ipynb
    
    Args:
        edges: Binary edge map from Canny detection
        rho: Distance resolution in pixels (smaller = more precise, slower)
        theta: Angle resolution in radians (np.pi/180 = 1 degree)
        threshold: Minimum number of intersections to detect a line
                   Higher = fewer, stronger lines
        min_line_length: Minimum length of line segment in pixels
                        Higher = only longer, straighter lines
        max_line_gap: Maximum gap between points to be considered same line
                     Lower = stricter, more broken lines
    
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


def filter_lines_by_slope(lines, min_slope=0.5, max_slope=2.0):
    """
    Filter detected lines by slope to keep only lane-like lines.
    
    Removes horizontal lines and overly steep lines.
    Separates left and right lane lines based on slope sign.
    
    Args:
        lines: Array of lines from detect_lines_hough()
        min_slope: Minimum absolute slope to keep (removes near-horizontal)
        max_slope: Maximum absolute slope to keep (removes near-vertical)
    
    Returns:
        left_lines: List of left lane line segments (negative slope)
        right_lines: List of right lane line segments (positive slope)
    """
    if lines is None:
        return [], []
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Skip vertical lines (avoid division by zero)
        if x2 - x1 == 0:
            continue
        
        # Calculate slope
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter by slope magnitude
        if abs(slope) < min_slope or abs(slope) > max_slope:
            continue
        
        # Separate by slope sign (left lanes negative, right lanes positive)
        if slope < 0:
            left_lines.append(line[0])
        else:
            right_lines.append(line[0])
    
    return left_lines, right_lines


def fit_lane_line(lines, frame_height):
    """
    Merge multiple line segments into ONE optimal straight line using linear regression.
    
    🎯 THIS IS THE KEY FUNCTION TO GET STRAIGHT, ACCURATE LANE LINES!
    Takes all the small line segments and fits them to a single best-fit line.
    
    Args:
        lines: List of line segments [[x1, y1, x2, y2], ...] or [[[x1, y1, x2, y2]], ...]
        frame_height: Height of the frame
        
    Returns:
        Fitted line as [x1, y1, x2, y2] or None if no lines
        Format: [x_top, y_top, x_bottom, y_bottom]
    """
    if not lines or len(lines) == 0:
        return None
    
    # Collect all points from all line segments
    x_points = []
    y_points = []
    
    for line in lines:
        # Handle both [[x1,y1,x2,y2]] and [x1,y1,x2,y2] formats
        if isinstance(line[0], (list, np.ndarray)) and len(line[0]) == 4:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line
        
        x_points.extend([x1, x2])
        y_points.extend([y1, y2])
    
    # Convert to numpy arrays
    x_points = np.array(x_points, dtype=np.float32)
    y_points = np.array(y_points, dtype=np.float32)
    
    # Fit polynomial (degree 1 = straight line)
    # Fit x as function of y (not y of x) because lines are nearly vertical
    poly = np.polyfit(y_points, x_points, 1)
    x_at_y = np.poly1d(poly)
    
    # Calculate line endpoints
    # Top: minimum y from all points
    y_top = int(np.min(y_points))
    x_top = int(x_at_y(y_top))
    
    # Bottom: frame height
    y_bottom = int(frame_height)
    x_bottom = int(x_at_y(y_bottom))
    
    return [x_top, y_top, x_bottom, y_bottom]


def extrapolate_line(line, y_start, y_end):
    """
    Extend a line segment to span between y_start and y_end.
    
    Args:
        line: Line segment [x1, y1, x2, y2]
        y_start: Starting y coordinate (typically frame bottom)
        y_end: Ending y coordinate (typically ROI top)
    
    Returns:
        Extended line [x_start, y_start, x_end, y_end]
    """
    x1, y1, x2, y2 = line
    
    # Handle vertical line (avoid division by zero)
    if x2 - x1 == 0:
        return [x1, y_start, x1, y_end]
    
    # Calculate slope and intercept: x = slope*y + intercept
    slope = (x2 - x1) / (y2 - y1) if y2 != y1 else 0
    intercept = x1 - slope * y1
    
    # Calculate x coordinates at y_start and y_end
    x_start = int(slope * y_start + intercept)
    x_end = int(slope * y_end + intercept)
    
    return [x_start, y_start, x_end, y_end]


# ============================================================
#                  5. VISUALIZATION HELPERS
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


def draw_lines(frame, lines, color=(255, 0, 0), thickness=3):
    """
    Draw detected line segments on frame.
    
    Args:
        frame: BGR image to draw on (will be modified)
        lines: List of line segments [[x1, y1, x2, y2], ...] or [x1, y1, x2, y2]
        color: BGR color tuple
        thickness: Line thickness
    
    Returns:
        frame: Frame with lines drawn
    """
    if lines is None or len(lines) == 0:
        return frame
    
    # Handle single line [x1, y1, x2, y2]
    if isinstance(lines, (list, np.ndarray)) and len(lines) == 4 and not isinstance(lines[0], (list, np.ndarray)):
        x1, y1, x2, y2 = lines
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        return frame
    
    # Handle multiple lines
    for line in lines:
        if len(line) == 4:  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = line
        else:  # [[x1, y1, x2, y2]]
            x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    
    return frame
