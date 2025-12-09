import cv2
import os

# ============================================================
#                PATHS & CONSTANTS
# ============================================================

RAW_VIDEO_PATH = "data/processed/highway_clip.mp4"   # THIS IS CORRECT
AUTO_FRAMES_DIR = "data/frames"                      # FIXED
SPECIAL_FRAMES_DIR = "data/frames_sample"            # FIXED

# Create directories if they don't exist
os.makedirs(AUTO_FRAMES_DIR, exist_ok=True)
os.makedirs(SPECIAL_FRAMES_DIR, exist_ok=True)

# ============================================================
#                1. Extract Frames Automatically
# ============================================================

def extract_frames_every(video_path, output_dir, interval=50):
    """
    Saves every X frames from the input video.
    interval = number of frames between each saved frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video:", video_path)
        return

    frame_id = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            out_path = f"{output_dir}/frame_{frame_id:05d}.png"
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_id += 1

    cap.release()
    print(f"[INFO] Extracted {saved} frames into {output_dir}")


# ============================================================
#         2. Extract Specific Frames (Important Moments)
# ============================================================

SPECIAL_TIMES = [
    1, 4, 8, 9, 10, 12, 27, 28, 29, 34,
    105, 107, 109, 110,
    143, 153,
    185, 186, 187, 188, 200, 231, 235,
    241, 242, 243, 244, 281, 282, 287, 288, 289, 290, 291, 294, 295
]

def extract_frames_by_times(video_path, output_dir, times_list):
    """
    Saves frames from specific timestamps given in seconds.
    times_list = list of times (in seconds) to extract.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    for t in times_list:
        frame_id = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()
        if ret:
            out_path = f"{output_dir}/frame_t{t}_f{frame_id}.png"
            cv2.imwrite(out_path, frame)
            print("[OK] Saved:", out_path)
        else:
            print("[ERROR] Could not extract frame at time:", t)

    cap.release()
    print(f"[INFO] Extracted {len(times_list)} special frames.")


# ============================================================
#                     MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=== PRE-PROCESSING STARTED ===")

    # 1. Automatic frame extraction
    extract_frames_every(
        video_path=RAW_VIDEO_PATH,
        output_dir=AUTO_FRAMES_DIR,
        interval=50
    )

    # 2. Extract important frames for lane-change scenarios
    extract_frames_by_times(
        video_path=RAW_VIDEO_PATH,
        output_dir=SPECIAL_FRAMES_DIR,
        times_list=SPECIAL_TIMES
    )

    print("=== PRE-PROCESSING FINISHED SUCCESSFULLY ===")