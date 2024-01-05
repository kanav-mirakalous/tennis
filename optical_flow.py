import cv2
import numpy as np

def find_euler_angles(rotation_vector):
    # Convert the rotation vector to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector[0])

    # Convert the rotation matrix to Euler angles
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] +  rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1] , rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def optical_flow_tracking(ROI, video_path, show=True):
    """take ROI from YOLO for corners and then track them using optical flow
    For added robustness we also use Shi-Tomasi corner detection to detect new corners within ROI
    ROI is a list of 4 values: x, y, w, h. current implementation is for entire image"""
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=10, qualityLevel=0.3, minDistance=3, blockSize=3)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_frame_roi = old_frame #[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
    old_gray = cv2.cvtColor(old_frame_roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes.
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Compute the average optical flow
        dx = np.mean(good_new[:, 0] - good_old[:, 0])
        dy = np.mean(good_new[:, 1] - good_old[:, 1])
        if abs(dx) > 1.0 or abs(dy) > 1.0:
            print('Camera motion detected')
            print(f'Camera motion: dx={dx}, dy={dy}')
        camera_movement=[dx, dy]
        
        # Compute the homography from the old points to the new points
        h, status = cv2.findHomography(good_old, good_new)
        K = np.eye(3)
        # Decompose the homography into rotation and translation components
        _, _, _, decomposed_rotation_vectors = cv2.decomposeHomographyMat(h, K)
        
        
        rotation_values=find_euler_angles(decomposed_rotation_vectors)
        print(f'Rotation values in radians (roll, pitch and yaw): {rotation_values}')

        
        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a =int(a)
            b = int(b)
            c = int(c)
            d = int(d)
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        cv2.waitKey(0)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        return rotation_values, camera_movement
    
        cv2.destroyAllWindows()
    cap.release()
    return -1, -1

def optical_flow_tracking_images(image1, image2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=10, qualityLevel=0.3, minDistance=3, blockSize=3)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Detect corners in the first image
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    dx = np.mean(good_new[:,0] - good_old[:,0])
    dy = np.mean(good_new[:,1] - good_old[:,1])

    return dx, dy

def optical_flow_tracking_frames(frame1, frame2):
    # Convert the frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect Shi-Tomasi corners in the first frame
    corners = cv2.goodFeaturesToTrack(gray1, mask=None, maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Select good points
    good_new = p1[st==1]
    good_old = corners[st==1]

    # Compute the movement in x and y direction
    dx = np.mean(good_new[:,0] - good_old[:,0])
    dy = np.mean(good_new[:,1] - good_old[:,1])

    # Find the homography
    H, _ = cv2.findHomography(good_old, good_new)

    # Decompose the homography into the camera matrix, rotation and translation vectors
    _, _, _,_,_, euler_angles = cv2.RQDecomp3x3(H[:3, :3])

    # Convert the Euler angles to radians
    euler_angles = [np.deg2rad(angle) for angle in euler_angles]

    return dx, dy, euler_angles

if __name__ == "__main__":

    #optical_flow_tracking([400, 300, 500, 400], 'images/AO.RodLaverArena1.mp4')
    video_path = "images/AO.RodLaverArena1.mp4"

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame1 = cap.read()

    while True:
        # Read the next frame
        ret, frame2 = cap.read()

        # Break the loop if we reached the end of the video
        if not ret:
            break

        # Call the function
        DX,DY,EA=optical_flow_tracking_frames(frame1, frame2)
        print(f'DX={DX}, DY={DY}, EA={EA}')

        # Update frame1 for the next iteration
        frame1 = frame2

    # Release the video file
    cap.release()

