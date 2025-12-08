from moviepy import VideoFileClip
clip = VideoFileClip("../data/raw/full_video_fixed.mp4")

start_time = 7*60 + 48   # 468 שניות
end_time   = 12*60 + 50  # 770 שניות

subclip = clip.subclipped(start_time, end_time)

# --- שלב 4: שמירה של הקטע לתיקייה processed ---
output_path = "../data/processed/highway_clip.mp4"
subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")

print("Saved trimmed clip to:", output_path)