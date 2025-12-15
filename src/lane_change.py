"""
Lane Change Detection and Temporal Smoothing
=============================================
This module provides temporal tracking and smoothing for lane detection.

Key features:
1. Temporal Smoothing - stabilizes lane lines by averaging over multiple frames
2. Lane Change Detection - identifies when vehicle is changing lanes


"""

from collections import deque
import numpy as np


class LaneChangeDetector:
    """
    Detect lane changes and smooth lane lines using temporal tracking.
    
    This class maintains a history of detected lane lines across frames
    and provides:
    - Smoothed lane lines (averaged over history)
    - Lane change detection (based on position variance)
    
    Args:
        history_size: Number of frames to keep in history (default: 7)
        x_threshold: Maximum x_bottom variance before flagging lane change (default: 150 pixels)
        min_valid: Minimum number of valid frames needed for detection (default: 4)
    """
    
    def __init__(self, history_size=7, x_threshold=150, min_valid=4):
        """
        Initialize lane change detector.
        
        Args:
            history_size: Number of frames to track (larger = smoother but slower response)
            x_threshold: Pixel threshold for lane change detection
            min_valid: Minimum frames needed for valid detection
        """
        self.history_size = history_size
        self.x_threshold = x_threshold
        self.min_valid = min_valid
        
        # History queues for left and right lanes
        self.left_history = deque(maxlen=history_size)
        self.right_history = deque(maxlen=history_size)
    
    def update(self, left_line, right_line):
        """
        Update detector with new frame and return smoothed lines + lane change flag.
        
        Args:
            left_line: Left lane line [x_top, y_top, x_bottom, y_bottom] or None
            right_line: Right lane line [x_top, y_top, x_bottom, y_bottom] or None
        
        Returns:
            tuple: (smoothed_left, smoothed_right, lane_change_detected)
                - smoothed_left: Smoothed left line or None
                - smoothed_right: Smoothed right line or None
                - lane_change_detected: True if lane change detected
        """
        # Add to history
        self.left_history.append(left_line)
        self.right_history.append(right_line)
        
        # Smooth lines
        smoothed_left = self._smooth_line(self.left_history)
        smoothed_right = self._smooth_line(self.right_history)
        
        # Detect lane change
        lane_change = self._detect_lane_change()
        
        return smoothed_left, smoothed_right, lane_change
    
    def _smooth_line(self, history):
        """
        Smooth a line by averaging over history.
        
        Args:
            history: Deque of line detections [x_top, y_top, x_bottom, y_bottom]
        
        Returns:
            Smoothed line [x_top, y_top, x_bottom, y_bottom] or None
        """
        # Filter out None values
        valid_lines = [line for line in history if line is not None]
        
        # Need minimum number of valid frames
        if len(valid_lines) < self.min_valid:
            # Return most recent valid line or None
            return valid_lines[-1] if valid_lines else None
        
        # Average all coordinates
        avg_line = np.mean(valid_lines, axis=0).astype(int)
        return list(avg_line)
    
    def _detect_lane_change(self):
        """
        Detect lane change based on x_bottom position variance.
        
        Returns:
            bool: True if lane change detected, False otherwise
        """
        # Check both left and right lanes
        for history in [self.left_history, self.right_history]:
            # Extract x_bottom positions (index 2 in [x_top, y_top, x_bottom, y_bottom])
            xs = [line[2] for line in history if line is not None]
            
            # Need minimum number of valid frames
            if len(xs) < self.min_valid:
                continue
            
            # Check if variance exceeds threshold
            variance = max(xs) - min(xs)
            if variance > self.x_threshold:
                return True
        
        return False
    
    def reset(self):
        """Clear all history (useful when starting new video segment)."""
        self.left_history.clear()
        self.right_history.clear()
