import cv2
import os

# Directory containing the videos
video_dir = 'videos'

# Iterate over each file in the video directory
for video_file in os.listdir(video_dir):
    # Construct the full path of the video file
    video_path = os.path.join(video_dir, video_file)

    # Create a directory for the frames of this video
    frame_dir = os.path.join(video_dir, os.path.splitext(video_file)[0])
    os.makedirs(frame_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f'Error: Unable to open video file {video_path}')
        continue

    frame_count = 0
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If the frame was not successfully read, then we have reached the end of the video
        if not ret:
            break

        # Save the frame as a JPEG image
        frame_file = os.path.join(frame_dir, f'frame{frame_count}.jpg')
        cv2.imwrite(frame_file, frame)

        frame_count += 1

    # Release the video file
    cap.release()