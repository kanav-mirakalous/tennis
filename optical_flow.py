import cv2
import numpy as np

def optical_flow_tracking(ROI, video_path):
    """take ROI from YOLO for corners and then track them using optical flow
    For added robustness we also use Shi-Tomasi corner detection to detect new corners within ROI
    ROI is a list of 4 values: x, y, w, h"""
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=10, qualityLevel=0.3, minDistance=3, blockSize=3)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_frame_roi = old_frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
    old_gray = cv2.cvtColor(old_frame_roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
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

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()