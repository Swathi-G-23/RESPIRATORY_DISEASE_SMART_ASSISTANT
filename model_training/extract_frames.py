import cv2
import os

# =========================================
# VIDEO FOLDER
# =========================================

video_folder = r"C:\Users\swath\OneDrive\Desktop\Smart Healthcare\dataset\videos"

# =========================================
# OUTPUT FRAMES FOLDER
# =========================================

output_folder = r"C:\Users\swath\OneDrive\Desktop\Smart Healthcare\dataset\frames"

os.makedirs(output_folder, exist_ok=True)

# =========================================
# START FRAME NUMBER
# =========================================

save_id = 1394

# =========================================
# FRAMES PER VIDEO
# =========================================

frames_per_video = 35

# =========================================
# LOOP THROUGH ALL VIDEOS
# =========================================

for video_name in os.listdir(video_folder):

    video_path = os.path.join(
        video_folder,
        video_name
    )

    # Skip non-video files
    if not video_name.endswith((
        ".mp4",
        ".avi",
        ".mov"
    )):
        continue

    print(f"\nProcessing: {video_name}")

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Step size
    step = max(
        total_frames // frames_per_video,
        1
    )

    frame_id = 0
    saved_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Save selected frames
        if frame_id % step == 0:

            filename = os.path.join(
                output_folder,
                f"frame_{save_id}.jpg"
            )

            cv2.imwrite(
                filename,
                frame
            )

            print(
                f"Saved: frame_{save_id}.jpg"
            )

            save_id += 1
            saved_count += 1

        # Stop after 35 frames
        if saved_count >= frames_per_video:
            break

        frame_id += 1

    cap.release()

print("\n✅ All pneumonia & asthma frames extracted successfully!")