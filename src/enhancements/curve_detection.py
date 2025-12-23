"""
Curve Detection System
======================
This module implements a complete curve detection system that:
1. Detects lane lines (left/right) during curved road driving
2. Fits polynomial curves (parabola) instead of straight lines
3. Smooths curves using temporal tracking
4. Visualizes curved lanes with polynomial curves

The system uses polynomial fitting (degree 2) to handle curved roads,
which is more accurate than straight-line fitting for curves.
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
    apply_color_threshold,
    apply_canny
)
from line_detection import (
    LaneChangeDetector
)


# ============================================================
#              ROI DEFINITION
# ============================================================

def get_curve_roi(frame, roi_params=None):
    """
    Get ROI for curve detection with adjusted bottom boundary.
    
    Args:
        frame: Input frame
        roi_params: ROI parameters dict (optional, uses defaults if None)
    
    Returns:
        roi_frame: Masked frame with curve ROI
        roi_points: Polygon points for visualization (with adjusted bottom)
    """
    h, w = frame.shape[:2]
    
    # Default ROI parameters (optimized for curves)
    if roi_params is None:
        roi_params = {
            "top_y_ratio": 0.61,       # Start ROI lower for better results
            "left_bottom_ratio": 0.07,  # Left edge at bottom of ROI
            "right_bottom_ratio": 0.92,  # Right edge at bottom of ROI
            "top_left_x_ratio": 0.44,  # Wider at top for curves
            "top_right_x_ratio": 0.56, # Wider at top for curves
        }
    
    # Extract bottom_y_ratio before passing to apply_roi_mask (it doesn't accept this parameter)
    bottom_y_ratio = roi_params.get("bottom_y_ratio", 0.93)  # ROI ends at 93% of image height (cut 7% from bottom - raised)
    
    # Create a copy of roi_params without bottom_y_ratio for apply_roi_mask
    roi_params_for_mask = {k: v for k, v in roi_params.items() if k != "bottom_y_ratio"}
    
    # Apply ROI mask (without bottom_y_ratio)
    roi_frame, roi_points = apply_roi_mask(frame, **roi_params_for_mask)
    
    # Adjust bottom y coordinates to raise bottom (cut more from bottom)
    # This avoids picking up lane markings from lanes below us
    bottom_y = int(h * bottom_y_ratio)
    roi_points[0][0][1] = bottom_y  # Bottom left y
    roi_points[0][1][1] = bottom_y  # Bottom right y
    
    # Recreate mask with adjusted points
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, roi_points, (255, 255, 255))
    roi_frame = cv2.bitwise_and(frame, mask)
    
    return roi_frame, roi_points


# ============================================================
#              CURVE FITTING - WORKING WITH PIXELS DIRECTLY
# ============================================================

def separate_pixels_simple(edges, roi_mask):
    """
    Simple pixel separation: Take all edge pixels in ROI and separate by x-position.
    
    No Hough, no regions, no complexity - just find all edge pixels and split left/right.
    Uses the center of the ROI (not the image center) for separation.
    
    Args:
        edges: Binary edge image (from Canny)
        roi_mask: Binary mask of ROI region
    
    Returns:
        left_points: List of (x, y) tuples for left lane pixels
        right_points: List of (x, y) tuples for right lane pixels
    """
    h, w = edges.shape
    
    # Apply ROI mask to edges
    roi_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    
    # Get all edge pixels in ROI
    edge_pixels = np.column_stack(np.where(roi_edges > 0))
    
    if len(edge_pixels) == 0:
        return [], []
    
    # Calculate ROI center (x-coordinate) from ROI mask
    # Find all pixels in ROI mask and calculate their mean x-coordinate
    roi_pixels = np.column_stack(np.where(roi_mask > 0))
    if len(roi_pixels) > 0:
        # roi_pixels is in (y, x) format, so we take column 1 for x
        roi_center_x = np.mean(roi_pixels[:, 1])
    else:
        # Fallback to image center if ROI is empty (shouldn't happen)
        roi_center_x = w // 2
    
    # Separate by x-position (split at ROI center)
    mid_x = int(roi_center_x)
    
    left_points = [(int(x), int(y)) for y, x in edge_pixels if x < mid_x]
    right_points = [(int(x), int(y)) for y, x in edge_pixels if x >= mid_x]
    
    return left_points, right_points


def separate_pixels_sliding_window(edges, roi_mask, top_ratio=0.3, window_width=100, min_pixels=50):
    """
    Separate pixels using sliding window in top region to find lane centers,
    then use these centers to separate all pixels.
    
    Algorithm:
    1. Use sliding window in top 30% of ROI to find left and right lane centers
    2. Use these centers to separate all pixels in ROI
    
    This works better for curves because it finds the actual lane positions
    in the top region (where lanes are more separated) and uses that to
    separate pixels throughout the ROI.
    
    Args:
        edges: Binary edge image (from Canny)
        roi_mask: Binary mask of ROI region
        top_ratio: Ratio of ROI height to use for sliding window (default: 0.3 = top 30%)
        window_width: Width of sliding window in pixels (default: 100)
        min_pixels: Minimum pixels in window to consider valid (default: 50)
    
    Returns:
        left_points: List of (x, y) tuples for left lane pixels
        right_points: List of (x, y) tuples for right lane pixels
    """
    h, w = edges.shape
    
    # Apply ROI mask to edges
    roi_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    
    # Get all edge pixels in ROI
    edge_pixels = np.column_stack(np.where(roi_edges > 0))
    
    if len(edge_pixels) == 0:
        return [], []
    
    # Convert to (x, y) format
    pixels_xy = np.array([(int(x), int(y)) for y, x in edge_pixels])
    
    # Get ROI boundaries from mask
    roi_y_coords = []
    for y in range(h):
        if np.any(roi_mask[y, :] > 0):
            roi_y_coords.append(y)
    
    if len(roi_y_coords) == 0:
        return [], []
    
    roi_top_y = min(roi_y_coords)
    roi_bottom_y = max(roi_y_coords)
    roi_height = roi_bottom_y - roi_top_y
    
    if roi_height == 0:
        return [], []
    
    # Top region for sliding window
    top_region_height = int(roi_height * top_ratio)
    top_region_bottom = roi_top_y + top_region_height
    
    # Get pixels in top region
    top_pixels = pixels_xy[(pixels_xy[:, 1] >= roi_top_y) & (pixels_xy[:, 1] < top_region_bottom)]
    
    if len(top_pixels) == 0:
        # Fallback to simple method if no pixels in top region
        return separate_pixels_simple(edges, roi_mask)
    
    # Sliding window to find lane centers in top region
    # Slide window from left to right
    left_lane_center = None
    right_lane_center = None
    left_lane_pixels_top = None
    right_lane_pixels_top = None
    
    # Find left lane (search from left side)
    for x_start in range(0, w - window_width, window_width // 2):
        x_end = min(x_start + window_width, w)
        window_pixels = top_pixels[(top_pixels[:, 0] >= x_start) & (top_pixels[:, 0] < x_end)]
        
        if len(window_pixels) >= min_pixels:
            left_lane_center = np.mean(window_pixels[:, 0])
            left_lane_pixels_top = window_pixels
            break
    
    # Find right lane (search from right side)
    for x_start in range(w - window_width, 0, -window_width // 2):
        x_end = min(x_start + window_width, w)
        window_pixels = top_pixels[(top_pixels[:, 0] >= x_start) & (top_pixels[:, 0] < x_end)]
        
        if len(window_pixels) >= min_pixels:
            right_lane_center = np.mean(window_pixels[:, 0])
            right_lane_pixels_top = window_pixels
            break
    
    # If we found both lanes, use midpoint to separate
    if left_lane_center is not None and right_lane_center is not None:
        # Use midpoint between lanes as separator
        separator_x = (left_lane_center + right_lane_center) / 2
    elif left_lane_center is not None:
        # Only left lane found, use it + offset
        separator_x = left_lane_center + 200  # Assume lane width ~200 pixels
    elif right_lane_center is not None:
        # Only right lane found, use it - offset
        separator_x = right_lane_center - 200  # Assume lane width ~200 pixels
    else:
        # No lanes found in top region, fallback to simple method
        return separate_pixels_simple(edges, roi_mask)
    
    # Separate all pixels using the separator
    left_points = pixels_xy[pixels_xy[:, 0] < separator_x].tolist()
    right_points = pixels_xy[pixels_xy[:, 0] >= separator_x].tolist()
    
    return left_points, right_points


def separate_pixels_clustering(edges, roi_mask):
    """
    Separate pixels using K-means clustering (2 clusters).
    
    This handles curves better than simple x-position split because it groups
    pixels by spatial proximity, not just x-coordinate.
    
    Algorithm:
    1. Collect all edge pixels in ROI
    2. Use K-means to cluster into 2 groups based on (x, y) position
    3. Determine which cluster is left and which is right by comparing mean x
    
    Args:
        edges: Binary edge image (from Canny)
        roi_mask: Binary mask of ROI region
    
    Returns:
        left_points: List of (x, y) tuples for left lane pixels
        right_points: List of (x, y) tuples for right lane pixels
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        # Fallback to simple method if sklearn not available
        return separate_pixels_simple(edges, roi_mask)
    
    h, w = edges.shape
    
    # Apply ROI mask to edges
    roi_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    
    # Get all edge pixels in ROI
    edge_pixels = np.column_stack(np.where(roi_edges > 0))
    
    if len(edge_pixels) == 0:
        return [], []
    
    # Convert to (x, y) format
    pixels_xy = np.array([(int(x), int(y)) for y, x in edge_pixels])
    
    if len(pixels_xy) < 2:
        return [], []
    
    # Use K-means to cluster into 2 groups
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_xy)
    
    # Separate by cluster
    cluster0_points = pixels_xy[labels == 0]
    cluster1_points = pixels_xy[labels == 1]
    
    # Determine which cluster is left and which is right
    # by comparing mean x position
    mean_x_0 = np.mean(cluster0_points[:, 0])
    mean_x_1 = np.mean(cluster1_points[:, 0])
    
    if mean_x_0 < mean_x_1:
        left_points = cluster0_points.tolist()
        right_points = cluster1_points.tolist()
    else:
        left_points = cluster1_points.tolist()
        right_points = cluster0_points.tolist()
    
    return left_points, right_points


def fit_curve_from_points(points, frame_height, outlier_threshold=50, max_iterations=2):
    """
    Fit polynomial curve directly from pixel points with iterative outlier removal.
    
    This is the core: x = a*y^2 + b*y + c
    
    Algorithm:
    1. Fit polynomial to all points
    2. Calculate distance of each point from the fitted curve
    3. Remove points that are too far (outliers)
    4. Fit again on remaining points
    5. Repeat until convergence or max iterations
    
    Args:
        points: List of (x, y) tuples
        frame_height: Height of the frame
        outlier_threshold: Maximum distance from curve to keep point (pixels, default: 50)
        max_iterations: Maximum number of iterations (default: 2)
    
    Returns:
        Polynomial coefficients [a, b, c] where x = a*y^2 + b*y + c, or None
        Also returns y_range [y_min, y_max]
    """
    if not points or len(points) < 3:
        return None, None
    
    # Require minimum points for reliable fitting
    min_points_for_fit = 8  # Need at least 8 points for reliable curve fitting
    if len(points) < min_points_for_fit:
        return None, None
    
    current_points = points.copy()
    
    for iteration in range(max_iterations):
        if len(current_points) < min_points_for_fit:
            break
        
        # Extract x and y coordinates
        x_points = np.array([p[0] for p in current_points], dtype=np.float32)
        y_points = np.array([p[1] for p in current_points], dtype=np.float32)
        
        # Calculate weights: give more weight to points at the top (smaller y)
        # This helps the polynomial fit better at the top edge (distant parts)
        y_min = np.min(y_points)
        y_max = np.max(y_points)
        y_range = y_max - y_min
        
        if y_range > 0:
            # Weight increases exponentially as we go up (smaller y)
            # Very strong emphasis on distant parts (top of ROI)
            # Using exponential with even higher alpha for much more weight to distant points
            alpha = 5.0  # Increased from 3.5 - gives much more weight to top (distant) points
            normalized_y = (y_points - y_min) / y_range  # 0 at top (distant), 1 at bottom (close)
            
            # Exponential weight: top points get ~4.0x weight, bottom points get ~1.0x weight
            # Very aggressive weighting for distant parts
            weights = 1.0 + 3.0 * np.exp(-alpha * normalized_y)  # ~4.0 at top, ~1.30 at bottom
            
            # Additional boost for the topmost 20% of points (most distant)
            top_20_percent_threshold = 0.2
            top_points_mask = normalized_y < top_20_percent_threshold
            weights[top_points_mask] *= 2.0  # Extra 100% boost (double) for most distant points (~8.0x total)
            
            # Normalize weights so they average to 1.0 (helps numerical stability)
            weights = weights / np.mean(weights)
        else:
            weights = np.ones_like(y_points)
        
        try:
            # Fit polynomial with weights: x = f(y) = a*y^2 + b*y + c
            poly = np.polyfit(y_points, x_points, deg=2, w=weights)
            poly_func = np.poly1d(poly)
            
            # Calculate distance of each point from the fitted curve
            x_predicted = poly_func(y_points)
            distances = np.abs(x_points - x_predicted)
            
            # Calculate quality metrics
            mean_error = np.mean(distances)
            max_error = np.max(distances)
            y_range = np.max(y_points) - np.min(y_points)
            
            # On last iteration, validate quality before returning
            if iteration == max_iterations - 1:
                # Quality checks:
                # 1. Mean error should be reasonable (less than 30 pixels)
                # 2. Max error should not be too high (less than 80 pixels)
                # 3. Y range should be sufficient (at least 100 pixels for reliable curve)
                # 4. Need enough points after filtering
                if (mean_error > 30 or max_error > 80 or y_range < 100 or 
                    len(current_points) < min_points_for_fit):
                    return None, None
                
                y_min = int(np.min(y_points))
                y_max = int(np.max(y_points))  # Use actual max of detected points, not frame_height
                return poly, [y_min, y_max]
            
            # Remove outliers (points too far from curve)
            # Use tighter threshold: calculate based on median + IQR for more aggressive filtering
            median_distance = np.median(distances)
            q1 = np.percentile(distances, 25)
            q3 = np.percentile(distances, 75)
            iqr = q3 - q1
            
            # Use tighter threshold: median + 1.2 * IQR (more aggressive than simple threshold)
            # But don't go below 70% of original threshold to avoid being too strict
            tight_threshold = min(outlier_threshold, max(median_distance + 1.2 * iqr, outlier_threshold * 0.7))
            
            mask = distances < tight_threshold
            filtered_points = [current_points[i] for i in range(len(current_points)) if mask[i]]
            
            # If we removed too many points, stop
            if len(filtered_points) < min_points_for_fit:
                # Validate quality before returning
                mean_error = np.mean(distances)
                max_error = np.max(distances)
                y_range = np.max(y_points) - np.min(y_points)
                if (mean_error > 30 or max_error > 80 or y_range < 100):
                    return None, None
                # Return the fit from current iteration
                y_min = int(np.min(y_points))
                y_max = int(np.max(y_points))  # Use actual max of detected points, not frame_height
                return poly, [y_min, y_max]
            
            # Update points for next iteration
            current_points = filtered_points
            
        except:
            # If polynomial fitting fails, don't fallback to linear
            # Linear fit is not reliable for curves
            return None, None
    
    # Final fit with remaining points (only if we have enough)
    if len(current_points) >= min_points_for_fit:
        x_points = np.array([p[0] for p in current_points], dtype=np.float32)
        y_points = np.array([p[1] for p in current_points], dtype=np.float32)
        try:
            poly = np.polyfit(y_points, x_points, deg=2)
            poly_func = np.poly1d(poly)
            x_predicted = poly_func(y_points)
            distances = np.abs(x_points - x_predicted)
            
            # Validate quality
            mean_error = np.mean(distances)
            max_error = np.max(distances)
            y_range = np.max(y_points) - np.min(y_points)
            
            if (mean_error > 30 or max_error > 80 or y_range < 100):
                return None, None
            
            y_min = int(np.min(y_points))
            y_max = int(frame_height)
            return poly, [y_min, y_max]
        except:
            return None, None
    
    return None, None


def extrapolate_curve(poly_coeffs, y_range, y_start, y_end):
    """
    Generate curve points for a polynomial curve between y_start and y_end.
    Extends the curve slightly beyond ROI boundaries for better visualization.
    
    Args:
        poly_coeffs: Polynomial coefficients [a, b, c] where x = a*y^2 + b*y + c
        y_range: Original y range [y_min, y_max] (not used, kept for compatibility)
        y_start: Starting y coordinate (typically ROI bottom - larger number)
        y_end: Ending y coordinate (typically ROI top - smaller number)
    
    Returns:
        curve_points: Array of points [[x1, y1], [x2, y2], ...] for the curve
    """
    if poly_coeffs is None:
        return None
    
    # Simply draw from top to bottom of ROI
    # y_start is bottom (larger), y_end is top (smaller)
    actual_y_top = min(y_start, y_end)    # Smaller value (top of ROI)
    actual_y_bottom = max(y_start, y_end)  # Larger value (bottom of ROI)
    
    # Extend very slightly beyond ROI boundaries (just 5 pixels up and down)
    extend_amount = 5  # Very small extension
    extended_y_top = max(0, actual_y_top - extend_amount)  # Extend upward (smaller y)
    extended_y_bottom = actual_y_bottom + extend_amount  # Extend downward (larger y)
    
    # Create polynomial function
    poly_func = np.poly1d(poly_coeffs)
    
    # Generate y values from extended top to extended bottom
    y_values = np.linspace(extended_y_top, extended_y_bottom, num=100)
    
    # Calculate x values
    x_values = poly_func(y_values)
    
    # Combine into points
    curve_points = np.array([[int(x), int(y)] for x, y in zip(x_values, y_values)], dtype=np.int32)
    
    return curve_points


# ============================================================
#              CURVE SMOOTHING
# ============================================================

class CurveDetector:
    """
    Simple temporal smoothing for curve coefficients with sudden change detection.
    
    The approach: store coefficients from last N frames and average them.
    Also detects sudden changes (outliers) that might indicate lane change or sliding.
    
    Args:
        history_size: Number of frames to keep in history (default: 3)
        min_valid: Minimum number of valid frames needed for smoothing (default: 2)
        sudden_change_threshold: Threshold for detecting sudden changes in coefficients (default: 0.5)
    """
    
    def __init__(self, history_size=2, min_valid=1, sudden_change_threshold=0.5):
        """
        Initialize curve detector.
        
        Args:
            history_size: Number of frames to track
            min_valid: Minimum frames needed for valid smoothing
            sudden_change_threshold: Threshold for detecting sudden changes (relative change in coefficients)
        """
        self.history_size = history_size
        self.min_valid = min_valid
        self.sudden_change_threshold = sudden_change_threshold
        
        # History queues for left and right curves (store polynomial coefficients)
        self.left_history = deque(maxlen=history_size)
        self.right_history = deque(maxlen=history_size)
        
        # Track sudden changes
        self.left_sudden_change = False
        self.right_sudden_change = False
    
    def update(self, left_curve, right_curve, y_range_left=None, y_range_right=None):
        """
        Update detector with new frame and return smoothed curves.
        
        Simple approach: average coefficients from last N frames.
        Also detects sudden changes that might indicate lane change or sliding.
        
        Args:
            left_curve: Left curve polynomial coefficients [a, b, c] or None
            right_curve: Right curve polynomial coefficients [a, b, c] or None
            y_range_left: Y range for left curve [y_min, y_max] or None
            y_range_right: Y range for right curve [y_min, y_max] or None
        
        Returns:
            tuple: (smoothed_left, smoothed_right, y_range_left, y_range_right, sudden_change_detected)
                - smoothed_left: Smoothed left curve coefficients or None
                - smoothed_right: Smoothed right curve coefficients or None
                - y_range_left: Y range for left curve
                - y_range_right: Y range for right curve
                - sudden_change_detected: True if sudden change detected in either curve
        """
        # Store curves with their y ranges
        left_data = (left_curve, y_range_left) if left_curve is not None else None
        right_data = (right_curve, y_range_right) if right_curve is not None else None
        
        # Smooth left curve and detect sudden changes
        smoothed_left, y_range_left, left_sudden = self._smooth_curve_simple(
            self.left_history, left_data
        )
        self.left_sudden_change = left_sudden
        
        # Smooth right curve and detect sudden changes
        smoothed_right, y_range_right, right_sudden = self._smooth_curve_simple(
            self.right_history, right_data
        )
        self.right_sudden_change = right_sudden
        
        # Sudden change detected if either curve changed suddenly
        sudden_change_detected = left_sudden or right_sudden
        
        return smoothed_left, smoothed_right, y_range_left, y_range_right, sudden_change_detected
    
    def _smooth_curve_simple(self, history, new_data):
        """
        Simple smoothing: average coefficients from history.
        Also detects sudden changes (outliers).
        
        This is the recommended approach - simple and effective.
        """
        # Filter out None values from history
        valid_curves = [data for data in history if data is not None]
        
        # Check for sudden change if we have previous curves and new data
        sudden_change = False
        if new_data is not None and len(valid_curves) >= 2:
            new_coeffs = new_data[0]
            # Compare with average of previous curves
            prev_coeffs = [c[0] for c in valid_curves]
            avg_prev_coeffs = np.mean(prev_coeffs, axis=0)
            
            # Calculate relative change
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_change = np.abs((new_coeffs - avg_prev_coeffs) / (np.abs(avg_prev_coeffs) + 1e-6))
            
            # Check if any coefficient changed significantly
            max_change = np.max(relative_change)
            if max_change > self.sudden_change_threshold:
                sudden_change = True
        
        # Add new curve to history (even if sudden change detected)
        history.append(new_data)
        
        # Update valid curves list
        valid_curves = [data for data in history if data is not None]
        
        # Need minimum number of valid frames
        if len(valid_curves) < self.min_valid:
            if valid_curves:
                return valid_curves[-1][0], valid_curves[-1][1], sudden_change
            return None, None, False
        
        # Simple average: a = mean(a_last), b = mean(b_last), c = mean(c_last)
        all_coeffs = [c[0] for c in valid_curves]
        avg_coeffs = np.mean(all_coeffs, axis=0).astype(float)
        
        # Average y ranges
        all_y_ranges = [c[1] for c in valid_curves]
        avg_y_range = [
            int(np.mean([r[0] for r in all_y_ranges])),
            int(np.mean([r[1] for r in all_y_ranges]))
        ]
        
        return list(avg_coeffs), avg_y_range, sudden_change
    
    def reset(self):
        """Clear all history (useful when starting new video segment)."""
        self.left_history.clear()
        self.right_history.clear()


# ============================================================
#              VISUALIZATION
# ============================================================

def draw_curves(frame, left_curve_coeffs, right_curve_coeffs, 
                 y_range_left, y_range_right, y_start, y_end):
    """
    Draw curved lane lines and polygon on frame.
    
    Args:
        frame: Frame to draw on
        left_curve_coeffs: Left curve polynomial coefficients [a, b, c] or None
        right_curve_coeffs: Right curve polynomial coefficients [a, b, c] or None
        y_range_left: Y range for left curve [y_min, y_max]
        y_range_right: Y range for right curve [y_min, y_max]
        y_start: Starting y coordinate (typically frame bottom)
        y_end: Ending y coordinate (typically ROI top)
    
    Returns:
        frame: Frame with curves drawn
    """
    if left_curve_coeffs is None or right_curve_coeffs is None:
        return frame
    
    # Generate curve points
    left_points = extrapolate_curve(left_curve_coeffs, y_range_left, y_start, y_end)
    right_points = extrapolate_curve(right_curve_coeffs, y_range_right, y_start, y_end)
    
    if left_points is None or right_points is None:
        return frame
    
    # Draw polygon (filled area between curves)
    # Combine left curve (forward) with right curve (backward) to form closed polygon
    pts = np.concatenate([left_points, right_points[::-1]])
    
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw curves as polylines
    cv2.polylines(frame, [left_points], False, (255, 0, 255), 3)
    cv2.polylines(frame, [right_points], False, (0, 255, 255), 3)
    
    return frame


# ============================================================
#              COMPLETE PIPELINE
# ============================================================

def process_frame_curve_detection(frame,
                                  curve_detector=None,
                                  lane_change_detector=None,
                                  roi_params=None,
                                  color_params=None,
                                  canny_params=None,
                                  enhancement_params=None,
                                  use_enhancement=True):
    """
    Complete pipeline for single frame with curve detection.
    
    Args:
        frame: Input BGR frame
        curve_detector: CurveDetector instance (optional) for temporal smoothing
        lane_change_detector: LaneChangeDetector instance (optional) for lane change detection
        roi_params: ROI parameters dict (optional, uses defaults if None)
        color_params: Color thresholding parameters dict (optional, only used if use_enhancement=False)
        canny_params: Canny edge detection parameters dict (optional)
        enhancement_params: Enhancement parameters dict (optional)
            - clahe_clip_limit: CLAHE clip limit (default: 2.0)
            - clahe_tile_grid_size: CLAHE tile grid size (default: (8, 8))
            - use_bilateral: Whether to use bilateral filter (default: True)
            - bilateral_d: Bilateral filter d parameter (default: 15)
            - bilateral_sigma_color: Bilateral filter sigma color (default: 5)
            - bilateral_sigma_space: Bilateral filter sigma space (default: 50)
            - threshold_value: Threshold value after enhancement (default: 112)
        use_enhancement: Whether to use enhancement before thresholding (default: True)
    
    Returns:
        result_frame: Frame with visualization
        curve_info: Dict with curve detection info (includes lane_change_detected)
    """
    h, w = frame.shape[:2]
    
    # Default parameters (optimized for curves)
    # roi_params defaults are handled by get_curve_roi() - no need to duplicate here
    
    if enhancement_params is None:
        enhancement_params = {
            "clahe_clip_limit": 1.3,
            "clahe_tile_grid_size": (8, 8),
            "use_bilateral": True,
            "bilateral_d": 11,
            "bilateral_sigma_color": 5,
            "bilateral_sigma_space": 25,
            "threshold_value": 111,  # Threshold value after enhancement
        }
    
    if color_params is None:
        color_params = {
            "white_lower": (0, 0, 190),
            "white_upper": (180, 20, 255),
            "yellow_lower": (15, 60, 90),
            "yellow_upper": (35, 255, 255),
            "morph_kernel_size": 3,
            "morph_open_iter": 1,
            "morph_close_iter": 2,
            "morph_dilate_iter": 2,
        }
    
    if canny_params is None:
        canny_params = {
            "low_threshold": 50,
            "high_threshold": 150,
            "blur_kernel": 5,
        }
    
    # Step 1: Apply ROI mask
    frame_roi, roi_points = get_curve_roi(frame, roi_params)
    
    # Step 2: Enhancement (CLAHE + Bilateral Filter) before thresholding
    if use_enhancement:
        # Convert to grayscale
        gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE enhancement
        clahe = cv2.createCLAHE(
            clipLimit=enhancement_params["clahe_clip_limit"],
            tileGridSize=enhancement_params["clahe_tile_grid_size"]
        )
        enhanced = clahe.apply(gray_roi)
        
        # Apply Bilateral Filter to reduce noise while preserving edges
        if enhancement_params["use_bilateral"]:
            smoothed = cv2.bilateralFilter(
                enhanced,
                enhancement_params["bilateral_d"],
                enhancement_params["bilateral_sigma_color"],
                enhancement_params["bilateral_sigma_space"]
            )
        else:
            smoothed = enhanced
        
        # Apply threshold
        _, thresholded = cv2.threshold(
            smoothed,
            enhancement_params["threshold_value"],
            255,
            cv2.THRESH_BINARY
        )
        
        # Step 3: Canny edge detection on thresholded image
        edges = apply_canny(thresholded, **canny_params)
    else:
        # Original pipeline: Color thresholding
        color_mask = apply_color_threshold(frame_roi, **color_params)
        # Step 3: Canny edge detection
        edges = apply_canny(color_mask, **canny_params)
    
    # Step 4: Pixel separation (simple x-position split)
    # Create ROI mask for pixel extraction
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, roi_points, 255)
    
    # Extract pixels using simple x-position split
    left_pixels, right_pixels = separate_pixels_simple(edges, roi_mask)
    
    # Step 5: Fit curves directly from pixels (polynomial degree 2)
    # This is the key: x = a*y^2 + b*y + c
    left_curve, y_range_left = fit_curve_from_points(left_pixels, h)
    right_curve, y_range_right = fit_curve_from_points(right_pixels, h)
    
    # Step 5.5: Validate curves - check minimum number of points
    # This is now handled in fit_curve_from_points, but keep as extra safety check
    min_points_required = 8  # Need at least 8 pixels for reliable curve fitting
    if left_curve is not None and len(left_pixels) < min_points_required:
        left_curve = None
    if right_curve is not None and len(right_pixels) < min_points_required:
        right_curve = None
    
    # Step 7: Detect lane change (before smoothing)
    # Get ROI boundaries for lane change detection
    roi_bottom_y = roi_points[0][0][1]
    roi_top_y = roi_points[0][3][1]
    
    lane_change_detected = False
    if lane_change_detector is not None:
        # Convert curves to line format for lane change detector
        # Use ROI boundaries (not full image height) for more accurate detection
        left_line_for_detector = None
        right_line_for_detector = None
        
        if left_curve is not None and y_range_left is not None:
            try:
                # Get x coordinates at top and bottom of ROI (not full image)
                poly_left = np.poly1d(left_curve)
                y_top = roi_top_y
                y_bottom = roi_bottom_y
                x_top_left = int(poly_left(y_top))
                x_bottom_left = int(poly_left(y_bottom))
                left_line_for_detector = [x_top_left, y_top, x_bottom_left, y_bottom]
            except:
                left_line_for_detector = None
        
        if right_curve is not None and y_range_right is not None:
            try:
                # Get x coordinates at top and bottom of ROI (not full image)
                poly_right = np.poly1d(right_curve)
                y_top = roi_top_y
                y_bottom = roi_bottom_y
                x_top_right = int(poly_right(y_top))
                x_bottom_right = int(poly_right(y_bottom))
                right_line_for_detector = [x_top_right, y_top, x_bottom_right, y_bottom]
            except:
                right_line_for_detector = None
        
        # Update lane change detector
        _, _, lane_change_detected = lane_change_detector.update(
            left_line_for_detector, right_line_for_detector
        )
    
    # Step 8: Apply smoothing if detector provided (only if not in lane change)
    sudden_change_detected = False
    if curve_detector is not None and not lane_change_detected:
        left_curve, right_curve, y_range_left, y_range_right, sudden_change_detected = curve_detector.update(
            left_curve, right_curve, y_range_left, y_range_right
        )
    
    # Step 9: Visualization
    result = frame.copy()
    
    # Only draw curves if not in lane change
    if not lane_change_detected and left_curve is not None and right_curve is not None:
        # Draw curves from ROI top to ROI bottom (use full ROI - bottom contains good information)
        extended_bottom_y = roi_bottom_y  # Use full ROI bottom
        extended_top_y = roi_top_y  # Top of ROI
        
        result = draw_curves(
            result, left_curve, right_curve,
            y_range_left, y_range_right,
            extended_bottom_y, extended_top_y
        )
    
    # Prepare info dict
    curve_info = {
        "left_curve": left_curve,
        "right_curve": right_curve,
        "detected": left_curve is not None and right_curve is not None,
        "lane_change_detected": lane_change_detected,
        "sudden_change_detected": sudden_change_detected
    }
    
    return result, curve_info


# ============================================================
#              VIDEO PROCESSING
# ============================================================

def process_video_curve(video_path, output_path, max_frames=None, start_frame=0,
                       curve_smoothing_history=3):
    """
    Process complete video with curve detection.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        max_frames: Limit number of frames (None = all)
        start_frame: Frame to start from (default: 0)
        curve_smoothing_history: Number of frames for curve smoothing (0 = no smoothing, default: 3)
    """
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
    print(f"VIDEO INFO - CURVE DETECTION")
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
        print(f"[SKIP] Skipping to frame {start_frame}\n")
    
    # Initialize curve detector if requested
    # Simple temporal smoothing: average coefficients from history
    # Smaller history for better sliding detection sensitivity
    curve_detector = None
    if curve_smoothing_history > 0:
        # Use smaller history (5 frames) for better sliding detection
        history_size = max(2, curve_smoothing_history)  # Reduced to 2 for faster response
        curve_detector = CurveDetector(
            history_size=history_size,
            min_valid=1  # Lower min_valid for faster response
        )
        print(f"[OK] Curve smoothing enabled: {history_size} frames history\n")
    
    # Initialize lane change detector (more sensitive for curves)
    lane_change_detector = LaneChangeDetector(
        history_size=15,
        missing_frames_threshold=3,  # More sensitive - detect lane change faster
        min_valid=3
    )
    print(f"[OK] Lane change detection enabled\n")
    
    # Lane change display timer (frames)
    lane_change_display_frames = 30  # Show warning for ~1 second
    lane_change_timer = 0
    
    # Video writer setup
    output_fps = max(10, fps // 2)
    
    # For MP4 files, try different codecs
    file_ext = os.path.splitext(output_path)[1].lower()
    
    if file_ext == '.mp4':
        # For MP4, try codecs that work with MP4 container
        codecs_to_try = [
            ('mp4v', 'mp4v'),  # MPEG-4 Part 2
            ('X264', 'H264'),  # H.264 (if available)
            ('XVID', 'XVID'),  # XVID (sometimes works)
        ]
    else:
        # For other formats (AVI, etc.)
        codecs_to_try = [
            ('XVID', 'XVID'),
            ('MJPG', 'MJPG'),
            ('DIVX', 'DIVX'),
        ]
    
    out = None
    used_codec = None
    
    for codec_name, fourcc_str in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(output_path, fourcc, output_fps, 
                                 (target_width, target_height) if resize_output else (width, height))
            if out.isOpened():
                used_codec = codec_name
                print(f"[OK] Video writer initialized with codec: {codec_name}")
                break
            else:
                out = None
        except Exception as e:
            print(f"[WARN] Failed to use codec {codec_name}: {e}")
            out = None
            continue
    
    if out is None or not out.isOpened():
        print("\n[ERROR] Cannot create video writer with any available codec")
        print("[INFO] Trying alternative: saving frames and using ffmpeg...")
        print("[INFO] This requires ffmpeg to be installed on your system")
        cap.release()
        return
    
    frame_count = start_frame
    detected_count = 0
    
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
        
        # Process frame with enhancement enabled by default
        result, curve_info = process_frame_curve_detection(
            frame, 
            curve_detector=curve_detector,
            lane_change_detector=lane_change_detector,
            use_enhancement=True  # Enable CLAHE + Bilateral enhancement
        )
        
        if curve_info["detected"]:
            detected_count += 1
        
        # Handle lane change detection
        lane_change_detected = curve_info.get("lane_change_detected", False)
        if lane_change_detected and not getattr(lane_change_detector, 'in_lane_change_active', False):
            lane_change_detector.in_lane_change_active = True
            lane_change_detector.reset()  # Clear history to prevent "pulling" old curves
            lane_change_timer = lane_change_display_frames  # Start display timer
        elif not lane_change_detected:
            lane_change_detector.in_lane_change_active = False
        
        # Decrement timer if active
        if lane_change_timer > 0:
            lane_change_timer -= 1
        
        # Add frame info
        cv2.putText(result, f"Frame: {frame_count}/{total_frames}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        
        # Status text
        if lane_change_detected or lane_change_timer > 0:
            status = "LANE CHANGE DETECTED"
            color = (0, 0, 255)  # Red
        elif curve_info["detected"]:
            status = "CURVES DETECTED"
            color = (0, 255, 0)  # Green
        else:
            status = "NO DETECTION"
            color = (0, 0, 255)  # Red
        
        cv2.putText(result, f"Status: {status}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        
        # Show lane change warning in center of screen
        if lane_change_detected or lane_change_timer > 0:
            text = "LANE CHANGE DETECTED!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 5
            text_color = (0, 0, 255)  # Red
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (width - text_width) // 2
            text_y = (height + text_height) // 2
            
            cv2.putText(result, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Resize output frame if needed
        if resize_output:
            result = cv2.resize(result, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Write frame
        out.write(result)
        
        # Progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames (Curves: {detected_count})")
    
    cap.release()
    out.release()
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Curve detections: {detected_count}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    video_path = os.path.join(project_root, 'data', 'processed', 'curve_clip.mp4')
    output_path = os.path.join(project_root, 'output', 'curve_detection_result.mp4')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Processing CURVE DETECTION VIDEO\n")
    process_video_curve(video_path, output_path, max_frames=None, start_frame=0)
    
    print(f"\nDone! You can now watch the result:")
    print(f"   {os.path.abspath(output_path)}")

