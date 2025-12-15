import cv2
import sys
import numpy as np
import os
from pipeline import (
    apply_roi_mask, 
    apply_color_threshold, 
    apply_canny,
    detect_lines_hough, 
    filter_lines_by_slope, 
    fit_lane_line,
    extrapolate_line, 
    draw_lines
)
from lane_change import LaneChangeDetector

def process_video(video_path, output_path, max_frames=None, start_frame=0):
    """
    Process video frame by frame and save result.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
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
    
    # Use ORIGINAL FPS to maintain same speed and duration
    output_fps = fps // 2 # ✅ Same as input = same speed!
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
    lane_detector = LaneChangeDetector(history_size=7, x_threshold=150, min_valid=4)
    
    frame_count = start_frame
    detected_count = 0
    lane_change_count = 0
    
    # Process every 2nd frame to reduce file size (still looks smooth)
    frame_skip = 2  # ✅ Skip every other frame = smaller file, still smooth
    
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
        
        # ============ PIPELINE ============
        time_sec = frame_count / fps
        
        # 3-PHASE ROI: Normal -> Uphill (sky issue) -> Close view
        if time_sec < 252:
            dynamic_top_y = 0.62
            top_left_x = 0.40
            top_right_x = 0.65
        elif 252 <= time_sec < 261:
            dynamic_top_y = 0.70
            top_left_x = 0.40
            top_right_x = 0.65
        else:
            dynamic_top_y = 0.60
            top_left_x = 0.40
            top_right_x = 0.65
        
        # 1. ROI with dynamic position and shape
        frame_roi, roi_pts = apply_roi_mask(
            frame, 
            top_y_ratio=dynamic_top_y,
            top_left_x_ratio=top_left_x,
            top_right_x_ratio=top_right_x
        )
        
        # 2. Color threshold
        color_mask = apply_color_threshold(frame_roi)
        
        # 3. Canny edges
        edges = apply_canny(color_mask)
        
        # 4. Detect lines
        lines = detect_lines_hough(edges, threshold=15, min_line_length=40, max_line_gap=20)
        
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
            
            # Calculate line extension - balanced length
            roi_bottom_y = roi_pts[0][0][1]
            roi_top_y = roi_pts[0][3][1]
            roi_height = roi_bottom_y - roi_top_y
            shorten_top = int(roi_height * 0.05)
            
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
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        status = "OK" if smoothed_left and smoothed_right else "MISSING"
        color = (0, 255, 0) if status == "OK" else (0, 0, 255)
        cv2.putText(result, f"Detection: {status}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add lane change warning
        if lane_change:
            cv2.putText(result, "LANE CHANGE DETECTED!", 
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
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
    
    # Process entire video
    print(f"🎬 Processing ENTIRE VIDEO\n")
    
    process_video(video_path, output_path, max_frames=None, start_frame=0)
    
    print(f"\n✅ Done! You can now watch the result:")
    print(f"   {os.path.abspath(output_path)}")