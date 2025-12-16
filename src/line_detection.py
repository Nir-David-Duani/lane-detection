"""
Line Detection, Lane Fitting, Temporal Smoothing, and Video Processing
========================================================================
This module handles everything after Hough line detection:
1. Filter lines by slope
2. Fit lane lines
3. Extrapolate lines
4. Lane change detection and temporal smoothing
5. Drawing and visualization
6. Video processing and output
"""

import cv2
import numpy as np
import os
from collections import deque
from pipeline import (
    apply_roi_mask,
    apply_color_threshold,
    apply_canny,
    detect_lines_hough
)


# ============================================================
#              LANE CHANGE DETECTION & SMOOTHING
# ============================================================

class LaneChangeDetector:
    """
    Detect lane changes and smooth lane lines using temporal tracking.
    
    This class maintains a history of detected lane lines across frames
    and provides:
    - Smoothed lane lines (averaged over history)
    - Lane change detection (based on missing lane detection period)
    
    Args:
        history_size: Number of frames to keep in history (default: 10)
        missing_frames_threshold: Number of consecutive frames with missing lanes to detect lane change (default: 10)
        min_valid: Minimum number of valid frames needed for detection (default: 4)
    """
    
    def __init__(self, history_size=10, missing_frames_threshold=10, min_valid=4):
        """
        Initialize lane change detector.
        
        Args:
            history_size: Number of frames to track (larger = smoother but slower response)
            missing_frames_threshold: Consecutive frames without both lanes to detect lane change
            min_valid: Minimum frames needed for valid detection
        """
        self.history_size = history_size
        self.missing_frames_threshold = missing_frames_threshold
        self.min_valid = min_valid
        
        # History queues for left and right lanes
        self.left_history = deque(maxlen=history_size)
        self.right_history = deque(maxlen=history_size)
        
        # Track consecutive frames with missing lanes
        self.consecutive_missing = 0
    
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
        Detect lane change based on consecutive frames with missing lane detection.
        
        Lane change is detected when both lanes are missing for a significant
        number of consecutive frames (indicating the vehicle is between lanes).
        
        Returns:
            bool: True if lane change detected, False otherwise
        """
        # Check if both lanes are missing in the most recent frame
        if len(self.left_history) == 0 or len(self.right_history) == 0:
            return False
        
        left_line = self.left_history[-1]
        right_line = self.right_history[-1]
        
        # Check if both lanes are missing
        if left_line is None or right_line is None:
            self.consecutive_missing += 1
        else:
            # Reset counter if both lanes are detected
            self.consecutive_missing = 0
        
        # Detect lane change if missing for threshold frames
        return self.consecutive_missing >= self.missing_frames_threshold
    
    def reset(self):
        """Clear all history (useful when starting new video segment)."""
        self.left_history.clear()
        self.right_history.clear()
        self.consecutive_missing = 0


# ============================================================
#              FILTER & FIT LANE LINES
# ============================================================

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
#              DRAWING & VISUALIZATION
# ============================================================

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


# ============================================================
#              VIDEO PROCESSING
# ============================================================

def process_video(video_path, output_path, max_frames=None, start_frame=0):
    """
    Run the complete lane detection pipeline on a video and save the result.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (with drawn lanes)
        max_frames: Limit number of frames (None = all)
        start_frame: Frame to start from (default: 0)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"{'='*60}")
    print(f"VIDEO INFO")
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
    
    # Video writer setup
    output_fps = fps // 2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    # Check if writer opened successfully
    if not out.isOpened():
        print("⚠️ XVID failed, trying mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        if not out.isOpened():
            print("❌ Fatal: Cannot create video writer")
            cap.release()
            return
    
    # Initialize lane change detector
    lane_detector = LaneChangeDetector(history_size=10, missing_frames_threshold=10, min_valid=4)
    
    frame_count = start_frame
    detected_count = 0
    lane_change_count = 0
    
    # Process every 2nd frame to reduce file size
    frame_skip = 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Limit frames if specified
        if max_frames and frame_count > max_frames:
            break
        
        # Skip frames for smaller file size
        if frame_count % frame_skip != 0:
            continue
        
        # ============ PIPELINE (from pipeline.py) ============
        
        # Fixed ROI parameters
        top_y_ratio = 0.6
        top_left_x = 0.40
        top_right_x = 0.65
        
        # 1. ROI
        frame_roi, roi_pts = apply_roi_mask(
            frame,
            top_y_ratio=top_y_ratio,
            top_left_x_ratio=top_left_x,
            top_right_x_ratio=top_right_x
        )
        
        # 2. Color threshold
        color_mask = apply_color_threshold(frame_roi)
        
        # 3. Canny edges
        edges = apply_canny(color_mask)
        
        # 4. Detect lines (Hough)
        lines = detect_lines_hough(edges, threshold=15, min_line_length=40, max_line_gap=20)
        
        # ============ LINE DETECTION & PROCESSING ============
        
        # 5. Filter by slope
        left_lines, right_lines = filter_lines_by_slope(lines, min_slope=0.5)
        
        # 6. Fit lane lines
        left_lane = fit_lane_line(left_lines, height)
        right_lane = fit_lane_line(right_lines, height)
        
        # 7. Smooth and detect lane change
        smoothed_left, smoothed_right, lane_change = lane_detector.update(left_lane, right_lane)
        
        if lane_change:
            lane_change_count += 1
        
        # 8. Extrapolate and draw (using SMOOTHED lines)
        result = frame.copy()
        if smoothed_left is not None and smoothed_right is not None:
            detected_count += 1
            
            # Calculate line extension
            roi_bottom_y = roi_pts[0][0][1]
            roi_top_y = roi_pts[0][3][1]
            roi_height = roi_bottom_y - roi_top_y
            # Shorten more at the top - lines won't extend too far forward
            shorten_top = int(roi_height * 0.12)  # Slightly longer lines
            
            extended_bottom_y = height
            extended_top_y = roi_top_y + shorten_top
            
            left_ext = extrapolate_line(smoothed_left, extended_bottom_y, extended_top_y)
            right_ext = extrapolate_line(smoothed_right, extended_bottom_y, extended_top_y)
            
            # Draw polygon
            pts = np.array([
                [left_ext[0], left_ext[1]],
                [left_ext[2], left_ext[3]],
                [right_ext[2], right_ext[3]],
                [right_ext[0], right_ext[1]]
            ], dtype=np.int32)
            
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
            
            # Draw lines
            draw_lines(result, left_ext, color=(255, 0, 255), thickness=3)
            draw_lines(result, right_ext, color=(0, 255, 255), thickness=3)
        
        # Add frame info
        cv2.putText(result, f"Frame: {frame_count}/{total_frames}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        
        status = "OK" if smoothed_left and smoothed_right else "MISSING"
        color = (0, 255, 0) if status == "OK" else (0, 0, 255)
        cv2.putText(result, f"Detection: {status}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        
        # Add lane change warning in center of screen in red (large)
        if lane_change:
            text = "LANE CHANGE DETECTED!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 5
            color = (0, 0, 255)  # Red
            
            # Get text size to center it
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (width - text_width) // 2
            text_y = (height + text_height) // 2
            
            cv2.putText(result, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Write frame
        out.write(result)
        
        # Progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames ({detected_count} detections)")
    
    cap.release()
    out.release()
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Successful detections: {detected_count}")
    print(f"Lane changes detected: {lane_change_count}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Set paths
    video_path = os.path.join('..', 'data', 'processed', 'highway_clip.mp4')
    output_path = os.path.join('..', 'output', 'test_pipeline.avi')
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("🎬 Processing ENTIRE VIDEO\n")
    process_video(video_path, output_path, max_frames=None, start_frame=0)
    
    print(f"\n✅ Done! You can now watch the result:")
    print(f"   {os.path.abspath(output_path)}")