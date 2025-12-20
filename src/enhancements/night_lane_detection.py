"""
Night-Time Lane Detection Enhancement
=====================================
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    apply_roi_mask,
    apply_canny
)

from line_detection import (
    extrapolate_line,
    draw_lines
)


############################################################
#              LANE CHANGE DETECTION & SMOOTHING           #
############################################################
from collections import deque

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


def process_video_night_robust(video_path, output_path, max_frames=None, start_frame=0):
    """
    Process video with night-time robust enhancement pipeline.
    
    Pipeline: ROI → Grayscale → CLAHE Enhancement → Bilateral Filter → Threshold → Canny → Hough
    
    This function works on both day and night videos by applying enhancement
    that improves visibility in low-light conditions without degrading
    daytime performance.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (with drawn lanes)
        max_frames: Limit number of frames (None = all)
        start_frame: Frame to start from (default: 0)
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
    
    print(f"{'='*60}")
    print(f"VIDEO INFO (Night-Robust Pipeline)")
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
        print(f"Skipping to frame {start_frame}\n")
    
    # Video writer setup
    # Output as mp4 with reduced size
    # Process every 2nd frame, so output FPS is halved
    frame_skip = 2
    output_fps = max(1, int(fps // frame_skip))  # Halve FPS, minimum 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Force .mp4 extension
    if not output_path.lower().endswith('.mp4'):
        output_path = os.path.splitext(output_path)[0] + '.mp4'
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    if not out.isOpened():
        print("Fatal: Cannot create mp4 video writer")
        cap.release()
        return
    
    # Initialize lane change detector
    # Reduced missing_frames_threshold to 5 for faster detection
    lane_detector = LaneChangeDetector(history_size=15, missing_frames_threshold=5, min_valid=4)
    
    # Initialize CLAHE for enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    
    # ROI parameters (synced with step_by_step and debug notebook)
    roi_params = {
        "top_y_ratio": 0.68,           # Top of trapezoid (step_by_step default)
        "left_bottom_ratio": 0.08,    # Left edge at bottom
        "right_bottom_ratio": 0.92,   # Right edge at bottom
        "top_left_x_ratio": 0.45,     # Left edge at top
        "top_right_x_ratio": 0.55,    # Right edge at top
    }

    # CLAHE parameters (synced with step_by_step)
    clahe_clip = 20
    clahe_grid = (2,2)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

    # Bilateral Filter parameters (synced with step_by_step)
    use_bilateral = True
    bilateral_d = 15
    bilateral_sigma_color = 5
    bilateral_sigma_space = 50

    # Threshold parameter (synced with step_by_step)
    threshold_value = 180

    # Canny parameters (unchanged)
    canny_params = {
        "low_threshold": 50,
        "high_threshold": 150,
        "blur_kernel": 5,
    }
    
    frame_count = start_frame
    detected_count = 0
    lane_change_count = 0
    processed_frames = 0  # Count of actually processed frames

    # Lane change display timer (frames)
    lane_change_display_frames = 30  # Show warning for 1 second at 30 FPS (will be scaled by output FPS)
    lane_change_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Limit frames if specified (check relative to start_frame)
        if max_frames and (frame_count - start_frame) > max_frames:
            break

        # Process only every 2nd frame
        if (frame_count - start_frame) % frame_skip != 0:
            continue

        processed_frames += 1

        # ============ ENHANCED PIPELINE (Notebook Style) ============

        # 1. Apply ROI mask
        frame_roi, roi_pts = apply_roi_mask(frame, **roi_params)

        # 2. Convert to grayscale
        gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

        # 3. Apply CLAHE enhancement (synced)
        enhanced = clahe.apply(gray_roi)

        # 3.5. Apply Bilateral Filter (synced)
        if use_bilateral:
            smoothed = cv2.bilateralFilter(enhanced, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
        else:
            smoothed = enhanced

        # 4. Apply threshold (synced)
        _, thresholded = cv2.threshold(smoothed, threshold_value, 255, cv2.THRESH_BINARY)

        # 5. Apply Canny edge detection
        edges = apply_canny(thresholded, **canny_params)

        # ============ LINE DETECTION & PROCESSING (Direct Polyfit) ============

        # 6. Detect lines using Hough Transform (like in line_detection.py)
        left_lane = None
        right_lane = None
        hough_rho = 1           # Distance resolution in pixels
        hough_theta = np.pi/180 # Angle resolution in radians
        hough_threshold = 40    # Minimum number of votes
        min_line_length = 20    # Minimum number of pixels making up a line
        max_line_gap = 20       # Maximum gap in pixels between connectable line segments
        lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        if lines is not None:
            mid_x = width // 2
            left_lines = []
            right_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Slope filtering: ignore nearly horizontal lines
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3:
                    continue
                # Classify as left or right
                if x1 < mid_x and x2 < mid_x:
                    left_lines.append([x1, y1, x2, y2])
                elif x1 >= mid_x and x2 >= mid_x:
                    right_lines.append([x1, y1, x2, y2])
            # Use mean of all detected lines for each side
            def average_line(lines):
                if not lines:
                    return None
                x1s, y1s, x2s, y2s = zip(*lines)
                return [int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))]
            left_lane = average_line(left_lines)
            right_lane = average_line(right_lines)

        # 8. Update history and detect lane change
        smoothed_left, smoothed_right, lane_change = lane_detector.update(left_lane, right_lane)

        # Improved Lane Change Logic with History Reset and display timer
        if lane_change and not getattr(lane_detector, 'in_lane_change_active', False):
            lane_detector.in_lane_change_active = True
            lane_change_count += 1
            lane_detector.reset() # Clear history to prevent "pulling" old lines
            lane_change_timer = lane_change_display_frames  # Start display timer
        elif not lane_change:
            lane_detector.in_lane_change_active = False
        # Decrement timer if active
        if lane_change_timer > 0:
            lane_change_timer -= 1


        # 9. Extrapolate and draw (using smoothed lines if available)
        result = frame.copy()

        # Determine which lines to draw (prefer smoothed)
        draw_left = smoothed_left if smoothed_left is not None else left_lane
        draw_right = smoothed_right if smoothed_right is not None else right_lane

        if draw_left is not None and draw_right is not None:
            roi_top_y = int(height * roi_params["top_y_ratio"])
            # Shorten the line: הארכה של 10% מגובה ה-ROI בלבד
            shorten_factor = 0.01 # 10% מהמרחק בין ROI ל-bottom
            extended_top_y = roi_top_y + int((height - roi_top_y) * shorten_factor)
            extended_bottom_y = height

            left_ext = extrapolate_line(draw_left, extended_bottom_y, extended_top_y)
            right_ext = extrapolate_line(draw_right, extended_bottom_y, extended_top_y)

            # Draw lane lines (Green, thinner)
            result = draw_lines(result, [left_ext], color=(0, 255, 0), thickness=2)
            result = draw_lines(result, [right_ext], color=(0, 255, 0), thickness=2)

            if left_ext and right_ext:
                pts = np.array([[left_ext[0], left_ext[1]], [left_ext[2], left_ext[3]],
                              [right_ext[2], right_ext[3]], [right_ext[0], right_ext[1]]], np.int32)
                overlay = result.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)

            detected_count += 1

        # Add frame info (like day video)
        cv2.putText(result, f"Frame: {frame_count}/{total_frames}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        # Show lane change warning for a short duration after detection (not every frame)
        if lane_change_timer > 0:
            text = "LANE CHANGE DETECTED!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 5
            color = (0, 0, 255)  # Red
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (width - text_width) // 2
            text_y = (height + text_height) // 2
            cv2.putText(result, text, (text_x, text_y), font, font_scale, color, thickness)

        # Write frame
        out.write(result)

        # Progress
        if processed_frames % 50 == 0:
            print(f"Processed {processed_frames} frames (frame {frame_count}) - {detected_count} detections")
    
    cap.release()
    out.release()
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Last frame number: {frame_count}")
    print(f"Successful detections: {detected_count}")
    print(f"Lane changes detected: {lane_change_count}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Video paths
    night_video_path = os.path.join(base_dir, 'data', 'processed', 'night_clip.mp4')
    
    # Output paths
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    night_output_path = os.path.join(output_dir, 'night_lane_detection_full.mp4')
    
    print("="*70)
    print("PROCESSING FULL NIGHT VIDEO")
    print("="*70)
    
    # Process the entire night video
    process_video_night_robust(
        night_video_path, 
        night_output_path, 
        max_frames=None, 
        start_frame=0
    )
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"Result saved to: {os.path.abspath(night_output_path)}")

