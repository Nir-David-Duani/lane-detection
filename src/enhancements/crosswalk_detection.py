"""
Crosswalk Detection System
==========================
This module implements a complete crosswalk detection system that:
1. Detects lane lines (left/right) during normal driving
2. Detects crosswalks using geometric pattern recognition
3. Highlights crosswalk area when detected
4. Temporarily disables lane line drawing when crosswalk is detected

The system uses two separate ROIs:
- ROI A: Lane detection (trapezoidal, lower-middle)
- ROI B: Crosswalk detection (rectangular, lower part)
"""

import cv2
import numpy as np
import sys
import os
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    apply_roi_mask,
    apply_canny,
    detect_lines_hough
)
from line_detection import (
    fit_lane_line,
    extrapolate_line,
    draw_lines
)


# ============================================================
#              ROI DEFINITION
# ============================================================

def get_lane_roi(frame):
    """
    ROI A - Lane detection region.
    Trapezoidal area in lower-middle part for diagonal/long lane lines.
    
    Args:
        frame: Input frame
    
    Returns:
        roi_frame: Masked frame with lane ROI
        roi_points: Polygon points for visualization
    """
    h, w = frame.shape[:2]
    
    roi_params = {
        "top_y_ratio": 0.65,
        "left_bottom_ratio": 0.15,
        "right_bottom_ratio": 0.78,
        "top_left_x_ratio": 0.37,
        "top_right_x_ratio": 0.50,
    }
    
    roi_frame, roi_points = apply_roi_mask(frame, **roi_params)
    return roi_frame, roi_points


def get_crosswalk_roi(frame, y_start_ratio=0.69, y_end_ratio=0.98, x_start_ratio=0.32, x_end_ratio=0.62):
    """
    ROI B - Crosswalk detection region.
    Rectangular area in lower part for dense patterns of short vertical lines.
    
    Args:
        frame: Input frame
        y_start_ratio: Vertical start position (0-1, default: 0.72)
        y_end_ratio: Vertical end position (0-1, default: 0.96)
        x_start_ratio: Horizontal start position (0-1, default: 0.30)
        x_end_ratio: Horizontal end position (0-1, default: 0.62)
    
    Returns:
        roi_frame: Masked frame with crosswalk ROI
        roi_points: Rectangle points for visualization
    """
    h, w = frame.shape[:2]
    
    y_start = int(h * y_start_ratio)
    y_end = int(h * y_end_ratio)
    x_start = int(w * x_start_ratio)
    x_end = int(w * x_end_ratio)
    
    # Create mask
    mask = np.zeros_like(frame)
    cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), (255, 255, 255), -1)
    roi_frame = cv2.bitwise_and(frame, mask)
    
    # ROI points for visualization
    roi_points = np.array([[
        (x_start, y_end),
        (x_end, y_end),
        (x_end, y_start),
        (x_start, y_start)
    ]], dtype=np.int32)
    
    return roi_frame, roi_points


# ============================================================
#              LINE EXTRACTION
# ============================================================

def extract_lines_from_roi(roi_gray, is_crosswalk_roi=False, 
                          threshold_value=None, canny_low=50, canny_high=150):
    """
    Extract lines using Threshold -> Canny -> Hough Transform.
    
    Args:
        roi_gray: ROI frame in grayscale
        is_crosswalk_roi: If True, use parameters optimized for crosswalk detection
        threshold_value: Threshold value (None = use default based on ROI type)
        canny_low: Canny low threshold
        canny_high: Canny high threshold
    
    Returns:
        lines: Array of line segments from Hough Transform
        edges: Edge map for visualization
    """
    # Ensure grayscale
    if len(roi_gray.shape) == 3:
        gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_gray
    
    # Thresholding
    if threshold_value is None:
        if is_crosswalk_roi:
            threshold_value = 200
        else:
            threshold_value = 210
    
    _, binary_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    if is_crosswalk_roi:
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    else:
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
    
    # Canny edge detection
    canny_params = {
        "low_threshold": canny_low,
        "high_threshold": canny_high,
        "blur_kernel": 7,
    }
    edges = apply_canny(binary_mask, **canny_params)
    
    # Post-process edges
    edge_kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel, iterations=2)
    edges = cv2.dilate(edges, edge_kernel, iterations=1)
    
    # Hough Transform
    if is_crosswalk_roi:
        hough_params = {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 10,
            "min_line_length": 15,
            "max_line_gap": 15,
        }
    else:
        hough_params = {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 15,
            "min_line_length": 40,
            "max_line_gap": 30,
        }
    
    lines = detect_lines_hough(edges, **hough_params)
    
    return lines, edges


# ============================================================
#              LANE LINE PROCESSING
# ============================================================

def filter_lines_by_slope_separate(lines, frame_width,
                                   left_min_slope=0.7, left_max_slope=1.6,
                                   right_min_slope=0.7, right_max_slope=1.6):
    """
    Filter detected lines by slope with separate ranges for left and right lanes.
    Also checks line position to ensure left lanes are on left side and right lanes on right side.
    
    Args:
        lines: Array of lines from detect_lines_hough()
        frame_width: Width of the frame (for position checking)
        left_min_slope: Minimum absolute slope for left lanes
        left_max_slope: Maximum absolute slope for left lanes
        right_min_slope: Minimum absolute slope for right lanes
        right_max_slope: Maximum absolute slope for right lanes
    
    Returns:
        left_lines: List of left lane line segments (negative slope, on left side)
        right_lines: List of right lane line segments (positive slope, on right side)
    """
    if lines is None:
        return [], []
    
    left_lines = []
    right_lines = []
    mid_x = frame_width // 2
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x2 - x1 == 0:
            continue
        
        slope = (y2 - y1) / (x2 - x1)
        abs_slope = abs(slope)
        
        # Filter out vertical lines
        if abs_slope > max(left_max_slope, right_max_slope):
            continue
        
        # Calculate line center x position
        line_center_x = (x1 + x2) / 2
        
        # Separate by slope sign AND position
        if slope < 0:  # Left lane (negative slope - tending left)
            # Check position: left lane should be on left side of frame
            if line_center_x < mid_x and left_min_slope <= abs_slope <= left_max_slope:
                left_lines.append(line[0])
        else:  # Right lane (positive slope - tending right)
            # Check position: right lane should be on right side of frame AND have positive slope
            if line_center_x >= mid_x and right_min_slope <= abs_slope <= right_max_slope and slope > 0:
                right_lines.append(line[0])
    
    return left_lines, right_lines


def process_lane_lines(lines, frame_height, frame_width,
                       left_min_slope=0.7, left_max_slope=1.6,
                       right_min_slope=0.7, right_max_slope=1.6):
    """
    Process and identify lane lines from ROI A.
    
    Args:
        lines: Lines from Hough Transform
        frame_height: Frame height for extrapolation
        frame_width: Frame width for position checking
        left_min_slope: Minimum absolute slope for left lanes (default: 0.7)
        left_max_slope: Maximum absolute slope for left lanes (default: 1.6)
        right_min_slope: Minimum absolute slope for right lanes (default: 0.7)
        right_max_slope: Maximum absolute slope for right lanes (default: 1.6)
    
    Returns:
        left_lane: Left lane line [x_top, y_top, x_bottom, y_bottom] or None
        right_lane: Right lane line [x_top, y_top, x_bottom, y_bottom] or None
    """
    if lines is None:
        return None, None
    
    # Filter by slope with separate ranges and position checking
    left_lines, right_lines = filter_lines_by_slope_separate(
        lines, frame_width,
        left_min_slope=left_min_slope,
        left_max_slope=left_max_slope,
        right_min_slope=right_min_slope,
        right_max_slope=right_max_slope
    )
    
    # Fit lane lines
    left_lane = fit_lane_line(left_lines, frame_height)
    right_lane = fit_lane_line(right_lines, frame_height)
    
    return left_lane, right_lane


# ============================================================
#              LANE SMOOTHING
# ============================================================

class LaneSmoother:
    """
    Smooth lane lines using temporal history with outlier detection.
    
    This class maintains a history of detected lane lines and provides:
    - Smoothed lane lines (averaged over history)
    - Outlier detection: if a line is significantly different, it checks if recent
      frames confirm a real change before updating
    
    Args:
        history_size: Number of frames to keep in history (default: 5)
        min_valid: Minimum number of valid frames needed for smoothing (default: 2)
        outlier_threshold: Maximum difference to consider a line an outlier (default: 100 pixels)
        confirmation_frames: Number of consecutive frames needed to confirm a change (default: 2)
    """
    
    def __init__(self, history_size=5, min_valid=2, outlier_threshold=100, confirmation_frames=2):
        """
        Initialize lane smoother.
        
        Args:
            history_size: Number of frames to track (larger = smoother but slower response)
            min_valid: Minimum frames needed for valid smoothing
            outlier_threshold: Maximum pixel difference to consider outlier
            confirmation_frames: Frames needed to confirm a real change
        """
        self.history_size = history_size
        self.min_valid = min_valid
        self.outlier_threshold = outlier_threshold
        self.confirmation_frames = confirmation_frames
        
        # History queues for left and right lanes
        self.left_history = deque(maxlen=history_size)
        self.right_history = deque(maxlen=history_size)
        
        # Track consecutive outlier frames
        self.left_outlier_count = 0
        self.right_outlier_count = 0
    
    def update(self, left_line, right_line):
        """
        Update smoother with new frame and return smoothed lines.
        
        Args:
            left_line: Left lane line [x_top, y_top, x_bottom, y_bottom] or None
            right_line: Right lane line [x_top, y_top, x_bottom, y_bottom] or None
        
        Returns:
            tuple: (smoothed_left, smoothed_right)
                - smoothed_left: Smoothed left line or None
                - smoothed_right: Smoothed right line or None
        """
        # Smooth left lane with outlier detection
        smoothed_left = self._smooth_line_with_outlier_detection(
            self.left_history, left_line, is_left=True
        )
        
        # Smooth right lane with outlier detection
        smoothed_right = self._smooth_line_with_outlier_detection(
            self.right_history, right_line, is_left=False
        )
        
        return smoothed_left, smoothed_right
    
    def _smooth_line_with_outlier_detection(self, history, new_line, is_left=True):
        """
        Smooth a line with outlier detection.
        
        Args:
            history: Deque of line detections
            new_line: New line detection [x_top, y_top, x_bottom, y_bottom] or None
            is_left: True if left lane, False if right lane
        
        Returns:
            Smoothed line [x_top, y_top, x_bottom, y_bottom] or None
        """
        # Add new line to history
        history.append(new_line)
        
        # Filter out None values
        valid_lines = [line for line in history if line is not None]
        
        # Need minimum number of valid frames
        if len(valid_lines) < self.min_valid:
            return valid_lines[-1] if valid_lines else None
        
        # Calculate average of previous lines (excluding current)
        if len(valid_lines) > 1:
            previous_lines = valid_lines[:-1]
            avg_line = np.mean(previous_lines, axis=0)
        else:
            avg_line = np.array(valid_lines[0])
        
        # Check if new line is an outlier
        if new_line is not None:
            # Calculate distance between new line and average
            distance = self._calculate_line_distance(new_line, avg_line)
            
            # Track outlier count
            outlier_count = self.left_outlier_count if is_left else self.right_outlier_count
            
            if distance > self.outlier_threshold:
                # This is an outlier - check if we've seen similar outliers recently
                outlier_count += 1
                
                if is_left:
                    self.left_outlier_count = outlier_count
                else:
                    self.right_outlier_count = outlier_count
                
                # If we've seen enough consecutive outliers, it's a real change
                if outlier_count >= self.confirmation_frames:
                    # Real change confirmed - use new line
                    if is_left:
                        self.left_outlier_count = 0
                    else:
                        self.right_outlier_count = 0
                    return list(new_line)
                else:
                    # Not confirmed yet - use average
                    return list(avg_line.astype(int))
            else:
                # Not an outlier - reset counter and use average including new line
                if is_left:
                    self.left_outlier_count = 0
                else:
                    self.right_outlier_count = 0
                # Average all lines including new one
                all_lines = valid_lines
                avg_all = np.mean(all_lines, axis=0).astype(int)
                return list(avg_all)
        else:
            # New line is None - use average of previous
            return list(avg_line.astype(int))
    
    def _calculate_line_distance(self, line1, line2):
        """
        Calculate average distance between two lines.
        
        Args:
            line1: Line [x_top, y_top, x_bottom, y_bottom]
            line2: Line [x_top, y_top, x_bottom, y_bottom] or numpy array
        
        Returns:
            Average pixel distance
        """
        if isinstance(line2, np.ndarray):
            line2 = line2.tolist()
        
        # Calculate distance at top and bottom points
        top_dist = np.sqrt((line1[0] - line2[0])**2 + (line1[1] - line2[1])**2)
        bottom_dist = np.sqrt((line1[2] - line2[2])**2 + (line1[3] - line2[3])**2)
        
        return (top_dist + bottom_dist) / 2
    
    def reset(self):
        """Clear all history (useful when starting new video segment)."""
        self.left_history.clear()
        self.right_history.clear()
        self.left_outlier_count = 0
        self.right_outlier_count = 0


# ============================================================
#              CROSSWALK PROCESSING
# ============================================================

def filter_horizontal_lines(lines, max_slope=0.1):
    """
    Filter lines to keep only horizontal lines.
    
    Args:
        lines: Lines from Hough Transform
        max_slope: Maximum absolute slope for horizontal lines (default: 0.1)
    
    Returns:
        horizontal_lines: List of horizontal line segments
    """
    if lines is None:
        return []
    
    horizontal_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if abs(x2 - x1) < 1:
            continue
        
        slope = abs((y2 - y1) / (x2 - x1))
        
        if slope <= max_slope:
            horizontal_lines.append(line[0])
    
    return horizontal_lines


def calculate_line_length(line):
    """Calculate length of a line segment."""
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def filter_crosswalk_candidates(horizontal_lines, min_length_ratio=0.55, min_distance=50):
    """
    Filter horizontal lines to find crosswalk candidates.
    
    Looks for two very long horizontal lines with minimum distance between them.
    
    Args:
        horizontal_lines: List of horizontal line segments
        min_length_ratio: Minimum line length as ratio of frame width (default: 0.5 = 50%)
        min_distance: Minimum distance between the two lines (default: 50 pixels)
    
    Returns:
        candidates: List of candidate lines (should be 2 long horizontal lines)
    """
    if len(horizontal_lines) < 2:
        return []
    
    # Calculate frame width from lines
    all_x = []
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        all_x.extend([x1, x2])
    frame_width = max(all_x) - min(all_x) if all_x else 1920
    min_length = int(frame_width * min_length_ratio)
    
    # Filter by length
    long_lines = []
    for line in horizontal_lines:
        length = calculate_line_length(line)
        if length >= min_length:
            long_lines.append(line)
    
    if len(long_lines) < 2:
        return []
    
    # Sort lines by y-coordinate
    long_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    
    # Group lines by similar y-coordinate
    line_groups = []
    current_group = [long_lines[0]]
    
    for line in long_lines[1:]:
        y_avg_current = (current_group[-1][1] + current_group[-1][3]) / 2
        y_avg_new = (line[1] + line[3]) / 2
        
        if abs(y_avg_new - y_avg_current) < 20:
            current_group.append(line)
        else:
            line_groups.append(current_group)
            current_group = [line]
    
    if current_group:
        line_groups.append(current_group)
    
    # Find the longest line in each group
    representative_lines = []
    for group in line_groups:
        longest = max(group, key=lambda l: calculate_line_length(l))
        representative_lines.append(longest)
    
    if len(representative_lines) < 2:
        return []
    
    # Sort by y-coordinate
    representative_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    
    # Check distance between top and bottom lines
    top_line = representative_lines[0]
    bottom_line = representative_lines[-1]
    
    top_y_center = (top_line[1] + top_line[3]) / 2
    bottom_y_center = (bottom_line[1] + bottom_line[3]) / 2
    distance = abs(bottom_y_center - top_y_center)
    
    if distance < min_distance:
        return []
    
    return [top_line, bottom_line]


def detect_crosswalk(horizontal_lines, min_lines=2, min_bbox_width=50, min_bbox_height=30):
    """
    Decide if crosswalk exists.
    
    Crosswalk is detected if:
    - At least min_lines horizontal long lines found
    - Bounding box meets minimum size requirements (if specified)
    
    Args:
        horizontal_lines: List of filtered horizontal line candidates
        min_lines: Minimum number of lines required (default: 2)
        min_bbox_width: Minimum bounding box width in pixels (default: 50)
        min_bbox_height: Minimum bounding box height in pixels (default: 30)
    
    Returns:
        bool: True if crosswalk detected, False otherwise
    """
    if len(horizontal_lines) < min_lines:
        return False
    
    # Check bounding box size requirements
    # Calculate bounding box
    all_x = []
    all_y = []
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        all_x.extend([x1, x2])
        all_y.extend([y1, y2])
    
    bbox_width = max(all_x) - min(all_x) if all_x else 0
    bbox_height = max(all_y) - min(all_y) if all_y else 0
    
    # Check minimum requirements
    if bbox_width < min_bbox_width:
        return False
    if bbox_height < min_bbox_height:
        return False
    
    return True


def get_crosswalk_bbox(crosswalk_lines):
    """
    Calculate bounding box around crosswalk lines.
    
    Args:
        crosswalk_lines: List of crosswalk line segments
    
    Returns:
        bbox: Tuple (x, y, width, height) or None
        points: Array of rectangle points for drawing or None
    """
    if len(crosswalk_lines) == 0:
        return None, None
    
    # Collect all endpoints
    all_x = []
    all_y = []
    for line in crosswalk_lines:
        x1, y1, x2, y2 = line
        all_x.extend([x1, x2])
        all_y.extend([y1, y2])
    
    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)
    
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)
    
    return bbox, points


# ============================================================
#              VISUALIZATION
# ============================================================

def draw_crosswalk(frame, bbox, points):
    """
    Draw crosswalk bounding box and text on frame.
    
    Args:
        frame: Frame to draw on
        bbox: Bounding box tuple (x, y, width, height)
        points: Array of rectangle points
    
    Returns:
        frame: Frame with crosswalk drawn
    """
    if bbox is None or points is None:
        return frame
    
    # Draw filled rectangle with transparency
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], (0, 255, 255))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw rectangle border
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
    
    # Draw text
    text = "CROSSWALK"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 255, 255)
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = bbox[0] + (bbox[2] - text_width) // 2
    text_y = bbox[1] - 10
    
    cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5),
                  (text_x + text_width + 5, text_y + baseline + 5),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return frame


def draw_lanes(frame, left_lane, right_lane, roi_points):
    """
    Draw lane lines and polygon on frame.
    
    Args:
        frame: Frame to draw on
        left_lane: Left lane line [x_top, y_top, x_bottom, y_bottom] or None
        right_lane: Right lane line [x_top, y_top, x_bottom, y_bottom] or None
        roi_points: ROI points for extrapolation
    
    Returns:
        frame: Frame with lanes drawn
    """
    if left_lane is None or right_lane is None:
        return frame
    
    h, w = frame.shape[:2]
    
    # Extrapolate lines
    roi_bottom_y = roi_points[0][0][1]
    roi_top_y = roi_points[0][3][1]
    roi_height = roi_bottom_y - roi_top_y
    shorten_top = int(roi_height * 0.12)
    
    extended_bottom_y = h
    extended_top_y = roi_top_y + shorten_top
    
    left_ext = extrapolate_line(left_lane, extended_bottom_y, extended_top_y)
    right_ext = extrapolate_line(right_lane, extended_bottom_y, extended_top_y)
    
    # Draw polygon
    pts = np.array([
        [left_ext[0], left_ext[1]],
        [left_ext[2], left_ext[3]],
        [right_ext[2], right_ext[3]],
        [right_ext[0], right_ext[1]]
    ], dtype=np.int32)
    
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw lines
    draw_lines(frame, left_ext, color=(255, 0, 255), thickness=3)
    draw_lines(frame, right_ext, color=(0, 255, 255), thickness=3)
    
    return frame


# ============================================================
#              COMPLETE PIPELINE
# ============================================================

def process_frame_crosswalk_detection(frame,
                                     lane_threshold=190, crosswalk_threshold=200,
                                     canny_low=50, canny_high=150,
                                     left_min_slope=0.7, left_max_slope=1.6,
                                     right_min_slope=0.7, right_max_slope=1.6,
                                     lane_smoother=None,
                                     min_crosswalk_width=50, min_crosswalk_height=30):
    """
    Complete pipeline for single frame.
    
    Args:
        frame: Input BGR frame
        lane_threshold: Threshold value for lane ROI (default: 190)
        crosswalk_threshold: Threshold value for crosswalk ROI (default: 200)
        canny_low: Canny low threshold (default: 50)
        canny_high: Canny high threshold (default: 150)
        left_min_slope: Minimum absolute slope for left lanes (default: 0.7)
        left_max_slope: Maximum absolute slope for left lanes (default: 1.6)
        right_min_slope: Minimum absolute slope for right lanes (default: 0.7)
        right_max_slope: Maximum absolute slope for right lanes (default: 1.6)
        lane_smoother: LaneSmoother instance (optional) for temporal smoothing
        min_crosswalk_width: Minimum bounding box width for crosswalk detection in pixels (default: 50)
        min_crosswalk_height: Minimum bounding box height for crosswalk detection in pixels (default: 30)
    
    Returns:
        result_frame: Frame with visualization
        crosswalk_detected: True if crosswalk detected
        lane_info: Dict with lane detection info
        crosswalk_info: Dict with crosswalk detection info
    """
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Get ROIs
    lane_roi_bgr, lane_roi_points = get_lane_roi(frame)
    crosswalk_roi_bgr, crosswalk_roi_points = get_crosswalk_roi(frame)
    
    # Convert ROIs to grayscale
    if len(lane_roi_bgr.shape) == 3:
        lane_roi_gray = cv2.cvtColor(lane_roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        lane_roi_gray = lane_roi_bgr
    
    if len(crosswalk_roi_bgr.shape) == 3:
        crosswalk_roi_gray = cv2.cvtColor(crosswalk_roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        crosswalk_roi_gray = crosswalk_roi_bgr
    
    # Step 3: Extract lines from each ROI
    lane_lines, lane_edges = extract_lines_from_roi(
        lane_roi_gray, 
        is_crosswalk_roi=False,
        threshold_value=lane_threshold,
        canny_low=canny_low,
        canny_high=canny_high
    )
    crosswalk_lines_raw, crosswalk_edges = extract_lines_from_roi(
        crosswalk_roi_gray, 
        is_crosswalk_roi=True,
        threshold_value=crosswalk_threshold,
        canny_low=canny_low,
        canny_high=canny_high
    )
    
    # Step 4: Process lane lines
    h, w = frame.shape[:2]
    left_lane, right_lane = process_lane_lines(
        lane_lines, h, w,
        left_min_slope=left_min_slope,
        left_max_slope=left_max_slope,
        right_min_slope=right_min_slope,
        right_max_slope=right_max_slope
    )
    
    # Apply smoothing if smoother is provided
    if lane_smoother is not None:
        left_lane, right_lane = lane_smoother.update(left_lane, right_lane)
    
    # Step 5: Process crosswalk lines
    horizontal_lines = filter_horizontal_lines(crosswalk_lines_raw)
    crosswalk_candidates = filter_crosswalk_candidates(horizontal_lines)
    
    # Step 6: Crosswalk decision
    crosswalk_detected = detect_crosswalk(
        crosswalk_candidates, 
        min_lines=2,
        min_bbox_width=min_crosswalk_width,
        min_bbox_height=min_crosswalk_height
    )
    
    # Step 7: Crosswalk bounding box
    crosswalk_bbox, crosswalk_points = get_crosswalk_bbox(crosswalk_candidates if crosswalk_detected else [])
    
    # Step 8: Visualization
    result = frame.copy()
    
    if crosswalk_detected:
        if crosswalk_bbox is not None:
            result = draw_crosswalk(result, crosswalk_bbox, crosswalk_points)
        else:
            # Draw text only if no bbox
            text = "CROSSWALK DETECTED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (0, 255, 255)
            
            h, w = frame.shape[:2]
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (w - text_width) // 2
            text_y = h - 30
            
            cv2.rectangle(result, (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + baseline + 5),
                         (0, 0, 0), -1)
            cv2.putText(result, text, (text_x, text_y), font, font_scale, color, thickness)
    else:
        # Draw lanes only when NO crosswalk detected
        if left_lane is not None and right_lane is not None:
            result = draw_lanes(result, left_lane, right_lane, lane_roi_points)
    
    # Prepare info dicts
    lane_info = {
        "left_lane": left_lane,
        "right_lane": right_lane,
        "detected": left_lane is not None and right_lane is not None
    }
    
    crosswalk_info = {
        "detected": crosswalk_detected,
        "bbox": crosswalk_bbox,
        "num_candidates": len(crosswalk_candidates)
    }
    
    return result, crosswalk_detected, lane_info, crosswalk_info


# ============================================================
#              VIDEO PROCESSING
# ============================================================

def process_video_crosswalk(video_path, output_path, max_frames=None, start_frame=0,
                           lane_smoothing_history=5):
    """
    Process complete video with crosswalk detection.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        max_frames: Limit number of frames (None = all)
        start_frame: Frame to start from (default: 0)
        lane_smoothing_history: Number of frames for lane smoothing (0 = no smoothing, default: 5)
    """
    import os
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Resize output for smaller file
    target_width = 1280
    target_height = 720
    resize_output = True if width > target_width or height > target_height else False
    
    print(f"{'='*60}")
    print(f"VIDEO INFO - CROSSWALK DETECTION")
    print(f"{'='*60}")
    print(f"Input: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.1f} seconds")
    if max_frames:
        print(f"Processing: First {max_frames} frames ({max_frames/fps:.1f} seconds)")
    else:
        print(f"Processing: ALL frames")
    print(f"{'='*60}\n")
    
    # Skip to start frame if specified
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"⏩ Skipping to frame {start_frame}\n")
    
    # Initialize lane smoother if requested
    lane_smoother = None
    if lane_smoothing_history > 0:
        lane_smoother = LaneSmoother(history_size=lane_smoothing_history)
        print(f"✅ Lane smoothing enabled: {lane_smoothing_history} frames\n")
    
    # Video writer setup
    output_fps = max(10, fps // 2)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, 
                         (target_width, target_height) if resize_output else (width, height))
    if not out.isOpened():
        print("Warning: H.264 not available, falling back to mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, 
                             (target_width, target_height) if resize_output else (width, height))
        if not out.isOpened():
            print("Fatal: Cannot create video writer (mp4v)")
            cap.release()
            return
    
    frame_count = start_frame
    crosswalk_detections = 0
    lane_detections = 0
    
    # Process every 2nd frame to reduce file size
    frame_skip = 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if max_frames and frame_count > max_frames:
            break
        
        if frame_count % frame_skip != 0:
            continue
        
        # Process frame
        result, crosswalk_detected, lane_info, crosswalk_info = process_frame_crosswalk_detection(
            frame, lane_smoother=lane_smoother
        )
        
        if crosswalk_detected:
            crosswalk_detections += 1
        if lane_info["detected"]:
            lane_detections += 1
        
        # Add frame info
        cv2.putText(result, f"Frame: {frame_count}/{total_frames}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        
        # Status text
        if crosswalk_detected:
            status = "CROSSWALK DETECTED"
            color = (0, 255, 255)
        elif lane_info["detected"]:
            status = "LANES DETECTED"
            color = (0, 255, 0)
        else:
            status = "NO DETECTION"
            color = (0, 0, 255)
        
        cv2.putText(result, f"Status: {status}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        
        # Resize output frame if needed
        if resize_output:
            result = cv2.resize(result, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Write frame
        out.write(result)
        
        # Progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames (Lanes: {lane_detections}, Crosswalks: {crosswalk_detections})")
    
    cap.release()
    out.release()
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Lane detections: {lane_detections}")
    print(f"Crosswalk detections: {crosswalk_detections}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    video_path = os.path.join(project_root, 'data', 'processed', 'crosswalk_clip.mp4')
    output_path = os.path.join(project_root, 'output', 'crosswalk_detection_result.mp4')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Processing CROSSWALK DETECTION VIDEO\n")
    process_video_crosswalk(video_path, output_path, max_frames=None, start_frame=0)
    
    print(f"\nDone! You can now watch the result:")
    print(f"   {os.path.abspath(output_path)}")
