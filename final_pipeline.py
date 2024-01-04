'''this file takes in the tennis video
   and outputs the coordinates of the court
   it also then produces the camera matrix
'''
from ultralytics import YOLO
import cv2
import numpy as np

#step 1 get the yolo model
# Load a model
model = YOLO('models/yolo_320.onnx')

# Initialize video capture
cap = cv2.VideoCapture('images/AO.RodLaverArena1.mp4')

# Initialize dictionaries for corners and poles
corners = {'upper_left': [], 'upper_right': [], 'lower_left': [], 'lower_right': []}
poles = {'left': [], 'right': []}

while True:
    # Read frame from video
    ret, frame = cap.read()

    # Break the loop if the video is over
    if not ret:
        break

    # Pass the frame through the model
    results = model(frame)

    # Process results
    for result in results:
        # Get the center of the bounding box
        center_x = (result.xmin + result.xmax) / 2
        center_y = (result.ymin + result.ymax) / 2

        # Check if the center is in the left or right half of the frame
        side = 'left' if center_x < frame.shape[1] / 2 else 'right'

        # Check if the label is 'corner' or 'pole' and store the bounding box
        if result.label == 'corner':
            # Check if the center is in the upper or lower half of the frame
            vertical_side = 'upper' if center_y < frame.shape[0] / 2 else 'lower'

            # Combine the side and vertical_side to get the quadrant
            quadrant = f'{vertical_side}_{side}'

            corners[quadrant].append(result)
        elif result.label == 'pole':
            poles[side].append(result)
    print(f"the corners are {corners}")
    print(f"the poles are {poles}")

    #reference on corrected frame
    corrected_frame = None

    # Define the destination points
    dst_pts = np.array([[0, 0], [frame.shape[1], 0], [0, frame.shape[0]], [frame.shape[1], frame.shape[0]]], dtype="float32")

    # Check for the four corners
    if 'upper_left' in corners and 'lower_left' in corners and 'upper_right' in corners and 'lower_right' in corners:
        # Define the source points
        src_pts = np.array([corners['upper_left'][0], corners['upper_right'][0], corners['lower_left'][0], corners['lower_right'][0]], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply the perspective transformation
        corrected_frame = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))

    # Check for the three corner combinations
    elif 'upper_left' in corners and 'lower_left' in corners and 'upper_right' in corners:
        # Define the source points
        src_pts = np.array([corners['upper_left'][0], corners['upper_right'][0], corners['lower_left'][0]], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getAffineTransform(src_pts, dst_pts[:3])

        # Apply the perspective transformation
        corrected_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    elif 'upper_left' in corners and 'lower_left' in corners and 'lower_right' in corners:
        # Define the source points
        src_pts = np.array([corners['upper_left'][0], corners['lower_left'][0], corners['lower_right'][0]], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getAffineTransform(src_pts, dst_pts[:3])

        # Apply the perspective transformation
        corrected_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    elif 'upper_right' in corners and 'lower_right' in corners and 'upper_left' in corners:
        # Define the source points
        src_pts = np.array([corners['upper_right'][0], corners['upper_left'][0], corners['lower_right'][0]], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getAffineTransform(src_pts, dst_pts[:3])

        # Apply the perspective transformation
        corrected_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    elif 'upper_right' in corners and 'lower_right' in corners and 'lower_left' in corners:
        # Define the source points
        src_pts = np.array([corners['upper_right'][0], corners['lower_right'][0], corners['lower_left'][0]], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getAffineTransform(src_pts, dst_pts[:3])

        # Apply the perspective transformation
        corrected_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    # Check for the combinations with poles
    elif 'upper_left' in corners and 'upper_right' in corners and 'left' in poles:
        # Define the source points
        src_pts = np.array([corners['upper_left'][0], corners['upper_right'][0], poles['left'][0]], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getAffineTransform(src_pts, dst_pts[:3])

        # Apply the perspective transformation
        corrected_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    elif 'lower_left' in corners and 'lower_right' in corners and 'left' in poles:
        # Define the source points
        src_pts = np.array([corners['lower_left'][0], corners['lower_right'][0], poles['left'][0]], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getAffineTransform(src_pts, dst_pts[:3])

        # Apply the perspective transformation
        corrected_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    elif 'upper_left' in corners and 'upper_right' in corners and 'right' in poles:
        # Define the source points
        src_pts = np.array([corners['upper_left'][0], corners['upper_right'][0], poles['right'][0]], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getAffineTransform(src_pts, dst_pts[:3])

        # Apply the perspective transformation
        corrected_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    elif 'lower_left' in corners and 'lower_right' in corners and 'right' in poles:
        # Define the source points
        src_pts = np.array([corners['lower_left'][0], corners['lower_right'][0], poles['right'][0]], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getAffineTransform(src_pts, dst_pts[:3])

        # Apply the perspective transformation
        corrected_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        # Release the video capture object
    # Define the source points
    src_pts = np.array([[0, 0], [corrected_frame.shape[1], 0], [0, corrected_frame.shape[0]], [corrected_frame.shape[1], corrected_frame.shape[0]]], dtype="float32")

    # Define the destination points
    dst_pts = np.array([corners['upper_left'][0], corners['upper_right'][0], corners['lower_left'][0], corners['lower_right'][0]], dtype="float32")

    # Find the homography
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Decompose the homography into the camera matrix, rotation and translation vectors
    _, K, R, T, _ = cv2.decomposeProjectionMatrix(H)

    # The camera position is the negative inverse of the translation vector
    camera_position = -np.linalg.inv(R).dot(T)

    # The camera rotation is the rotation matrix
    camera_rotation = R

    # Convert the rotation matrix to Euler angles
    _, _, _, _, _, _, euler_angles = cv2.RQDecomp3x3(camera_rotation)

    # Convert the Euler angles to radians
    euler_angles = [np.deg2rad(angle) for angle in euler_angles]

    print(f'Camera position: {camera_position}')
    print(f'Camera rotation: {camera_rotation}')
    print(f'Camera Euler angles: {euler_angles}')

cap.release()