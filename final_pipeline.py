'''this file takes in the tennis video
   and outputs the coordinates of the court
   it also then produces the camera matrix
'''
from ultralytics import YOLO
import cv2
import numpy as np
import time
import math
import random
from optical_flow import optical_flow_tracking_frames

output_camera_struct=None 
output_initial_camera_struct=None
output_mobile_struct=None

def fill_missing_elements(detected_elements, calibrated_elements):
    '''for filling in missing elements'''
    filled_elements = detected_elements.copy()
    for element in calibrated_elements:
        # Check if the element is in the detected elements
        if element not in detected_elements:
            # If not, add the calibrated position
            filled_elements[index]=calibrated_elements[element]
    return filled_elements

def process_file_for_camera_angles(video_path):

    #step 1 get the yolo model
    # Load a model
    model = YOLO('models/yolo_320.onnx')
    initial_camera_struct={}
    initial_camera_struct['rotation_vector']=[]
    initial_camera_struct['translation_vector']=[]
    initial_camera_struct['yaw']=0
    initial_camera_struct['pitch']=0
    initial_camera_struct['roll']=0
    # Initialize video capture
    # Initialize dictionaries for corners and poles
    corners = {'upper_left': [], 'upper_right': [], 'lower_left': [], 'lower_right': []}
    poles = {'left': [], 'right': []}
    corners_frame = {'upper_left': [], 'upper_right': [], 'lower_left': [], 'lower_right': []}
    poles_frame = {'left': [], 'right': []}
    calibrated_corners = {'upper_left': [], 'upper_right': [], 'lower_left': [], 'lower_right': []}
    calibrated_poles = {'left': [], 'right': []}
    current_calibrated_corners = {'upper_left': [], 'upper_right': [], 'lower_left': [], 'lower_right': []}
    current_calibrated_poles = {'left': [], 'right': []}
    # Initialize previous frame
    prev_frame = None
    # Initialize previous corners and poles
    prev_corners = None
    prev_poles = None
    n=0
    initial_camera_struct, status=find_camera_angles_calibrated(video_path, model, corners, poles, calibrated_corners, calibrated_poles, prev_frame, prev_corners, prev_poles, n, initial_camera_struct)
    if status:
        print("calibration achieved")
        #we now have the camera parameters yaw pitch roll and translation vector
        #the next flow is as follows. We take the video and pass it through the yolo model. now if we get atleast three points 
        cap2 = cv2.VideoCapture(video_path)
        current_calibrated_corners=calibrated_corners
        current_calibrated_poles=calibrated_poles
        prev_frame = None
        n=0
        while True:
            # Read frame from video
            ret, frame = cap2.read()
            # Break the loop if the video is over
            print(f"the frame number is {n}")
          
            if not ret:
                break
            #Pass the frame through the model
            results = model(frame, imgsz=320, task='detect', save=True)
            result=results[0]
            cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
            probs = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            labels=[]
            for cl in cls:
                if cl==0:
                    labels.append('corner')
                else:
                    labels.append('pole')
            box_centres=[]
            for box in boxes:
                x1, y1, x2, y2 = box
                box_centres.append([int((x1+x2)/2), int((y1+y2)/2)])
            median_x = np.median([box_centre[0] for box_centre in box_centres])
            median_y = np.median([box_centre[1] for box_centre in box_centres])
            print(f"detected labels are {labels}")
            print(f"detected box centres are {box_centres}")
            
            for box_centre, label in zip(box_centres, labels):
                side = 'left' if box_centre[0] < median_x else 'right'
                if label == 'corner':
                    # Check if the center is in the upper or lower half of the frame
                    vertical_side = 'upper' if box_centre[1] < frame.shape[0] / 2 else 'lower'
                    # Combine the side and vertical_side to get the quadrant
                    quadrant = f'{vertical_side}_{side}'
                    corners_frame[quadrant]=(box_centre)
                elif label == 'pole':
                    poles_frame[side]=(box_centre)
            
            for key in corners_frame.keys():
                if not corners_frame[key]:  # if the list is empty
                    corners_frame[key] = calibrated_corners[key]
                else:
                    for i in range(len(corners_frame[key])):
                        if not corners_frame[key][i]:  # if the element is None or 0 or False
                            corners_frame[key][i] = calibrated_corners[key][i]

            for key in poles_frame.keys():
                if not poles_frame[key]:  # if the list is empty
                    poles_frame[key] = calibrated_poles[key]
                else:
                    for i in range(len(poles_frame[key])):
                        if not poles_frame[key][i]:  # if the element is None or 0 or False
                            poles_frame[key][i] = calibrated_poles[key][i]
            for key in corners_frame:
                # Flatten the list if it's a list of lists
                if corners_frame[key] and isinstance(corners_frame[key][0], list):
                    corners_frame[key] = [item for sublist in corners_frame[key] for item in sublist]
            for key in poles_frame:
                # Flatten the list if it's a list of lists
                if poles_frame[key] and isinstance(poles_frame[key][0], list):
                    poles_frame[key] = [item for sublist in poles_frame[key] for item in sublist]

            print(f"filled corners are {corners_frame}")
            print(f"filled poles are {poles_frame}")
            
            #update the current calibrated corners and poles
            current_calibrated_corners.update(corners_frame)
            current_calibrated_poles.update(poles_frame)
            print(f"current calibrated corners are {current_calibrated_corners}")
            
            #finding non-empty corners and poles
            width=frame.shape[0]
            height=frame.shape[1]
            # Define the destination points
            dst_pts = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]], dtype='float32')

            # Get the source points from the calibrated corners
            print(f"current calibrated corners['upper left'] are {current_calibrated_corners['upper_left']}")
            list_of_points=[[calibrated_corners['upper_left'][0],calibrated_corners['upper_left'][1]], calibrated_corners['upper_right'], 
                                calibrated_corners['lower_left'], calibrated_corners['lower_right']]
            src_pts = np.array(list_of_points, dtype='float32')

            # Calculate the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Apply the perspective transformation to the frame
            corrected_frame = cv2.warpPerspective(frame, M, (width, height))

            # Save the corrected frame
            cv2.imwrite('corrected_frame.png', corrected_frame)

            # 3D points in the world coordinate system
            object_points = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]], dtype='float32')

            # Corresponding 2D points in the image. You need to replace these with the actual pixel coordinates.
            image_points = np.array([current_calibrated_corners['upper_left'], current_calibrated_corners['lower_right'], current_calibrated_corners['upper_right'], current_calibrated_corners['lower_left']], dtype='float32')

            # Camera matrix (assuming focal length = 1 and center = (0, 0))
            camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32')

            # Distortion coefficients (assuming no distortion)
            dist_coeffs = np.zeros((4, 1))

            # Solve the PnP problem to find the camera pose
            _, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

            # Convert the rotation vector to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, roll = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))[3:6]
            pitch = pitch[0]
            yaw = yaw[0]
            roll = roll[0]
            print(f" the rotation vector is {rotation_vector}")
            print(f" the translation vector is {translation_vector}")
            print(f" the pitch is {pitch}")
            print(f" the yaw is {yaw}")
            print(f" the roll is {roll}")
            camera_struct={}
            camera_struct['rotation_vector']=rotation_vector
            camera_struct['translation_vector']=translation_vector
            camera_struct['yaw']=yaw
            camera_struct['pitch']=pitch
            camera_struct['roll']=roll
            mobile_struct={}
            if n>0:
                dx_mobile, dy_mobile, euler_angles_mobile=optical_flow_tracking_frames(prev_frame, frame)
                mobile_struct['dx']=dx_mobile
                mobile_struct['dy']=dy_mobile
                mobile_struct['euler_angles']=euler_angles_mobile
                n=n+1
            else:
                n=n+1
            prev_frame=frame
            global output_camera_struct, output_initial_camera_struct, output_mobile_struct
            output_camera_struct=camera_struct
            output_initial_camera_struct=initial_camera_struct
            output_mobile_struct=mobile_struct
            access_structs()
        cap2.release()
    else:
        print("calibration failed...try again")
        return False

def access_structs():
    global output_camera_struct, output_initial_camera_struct, output_mobile_struct
    print(f"the camera struct is {output_camera_struct}")
    print(f"the initial camera struct is {output_initial_camera_struct}")
    print(f"the mobile struct is {output_mobile_struct}")
    return output_camera_struct, output_initial_camera_struct, output_mobile_struct


def calibration(calibrated_corners, calibrated_poles, new_corners, new_poles):
    # Iterate over new corners and poles
    for key in new_corners:
        # If old corner list for this key is empty and new corner list for this key is not empty
        if not calibrated_corners[key] and new_corners[key]:
            calibrated_corners[key] = new_corners[key][0]

    for key in new_poles:
        # If old pole list for this key is empty and new pole list for this key is not empty
        if not calibrated_poles[key] and new_poles[key]:
            calibrated_poles[key] = new_poles[key][0]

    # Check if all corners and poles are populated
    if all(calibrated_corners.values()) and all(calibrated_poles.values()):
        return calibrated_corners, calibrated_poles, True

    return calibrated_corners, calibrated_poles, False

def find_camera_angles_calibrated(video_path, model, corners, poles, calibrated_corners, calibrated_poles, prev_frame, prev_corners, prev_poles, n, initial_camera_struct):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    while True:
        # Read frame from video
        ret, frame = cap.read()
        n=n+1

        # If the previous frame is not None, calculate optical flow
        # if prev_frame is not None:
        #     dx, dy, _ = optical_flow_tracking_frames(prev_frame, frame)

        # Break the loop if the video is over
        if not ret:
            break

        # Pass the frame through the model
        results = model(frame, imgsz=320, task='detect', save=True)
        result=results[0]
        
        cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
        probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
        boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)
        labels=[]
        for cl in cls:
            if cl==0:
                labels.append('corner')
            else:
                labels.append('pole')
        
        box_centres=[]
        for box in boxes:
            x1, y1, x2, y2 = box
            box_centres.append([int((x1+x2)/2), int((y1+y2)/2)])
        

        median_x = np.median([box_centre[0] for box_centre in box_centres])
        median_y = np.median([box_centre[1] for box_centre in box_centres])
        
        for box_centre, label in zip(box_centres, labels):
            side = 'left' if box_centre[0] < median_x else 'right'
            if label == 'corner':
                # Check if the center is in the upper or lower half of the frame
                vertical_side = 'upper' if box_centre[1] < frame.shape[0] / 2 else 'lower'
                # Combine the side and vertical_side to get the quadrant
                quadrant = f'{vertical_side}_{side}'
                
                corners[quadrant].append(box_centre)
            elif label == 'pole':
                poles[side].append(box_centre)
        
        calibrated_corners, calibrated_poles, status=calibration(calibrated_corners, calibrated_poles, corners, poles)
        if n==300 and status is not True:
            print("calibration failed")
            print("hold the camera steady to capture the entire court and try again")
            return initial_camera_struct, False
            
        elif status is True:
            print("calibration successful")
            print("calibrated corners are:")
            print(calibrated_corners)
            print("calibrated poles are:")
            print(calibrated_poles)
            #we now have calibrated corners and poles. we can now calculate the camera matrix
            # Define the source points
            # Define the destination points
            width=frame.shape[0]
            height=frame.shape[1]
            # Define the destination points
            dst_pts = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]], dtype='float32')

            # Get the source points from the calibrated corners
            src_pts = np.array([calibrated_corners['upper_left'], calibrated_corners['upper_right'], 
                                calibrated_corners['lower_left'], calibrated_corners['lower_right']], dtype='float32')

            # Calculate the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Apply the perspective transformation to the frame
            corrected_frame = cv2.warpPerspective(frame, M, (width, height))

            # Save the corrected frame
            cv2.imwrite('corrected_frame.png', corrected_frame)

            # 3D points in the world coordinate system
            object_points = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]], dtype='float32')

            # Corresponding 2D points in the image. You need to replace these with the actual pixel coordinates.
            image_points = np.array([calibrated_corners['upper_left'], calibrated_corners['lower_right'], calibrated_corners['upper_right'], calibrated_corners['lower_left']], dtype='float32')

            # Camera matrix (assuming focal length = 1 and center = (0, 0))
            camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32')

            # Distortion coefficients (assuming no distortion)
            dist_coeffs = np.zeros((4, 1))

            # Solve the PnP problem to find the camera pose
            _, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

            # Convert the rotation vector to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, roll = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))[3:6]
            pitch = pitch[0]
            yaw = yaw[0]
            roll = roll[0]
            print(f" the rotation vector is {rotation_vector}")
            print(f" the translation vector is {translation_vector}")
            print(f" the pitch is {pitch}")
            print(f" the yaw is {yaw}")
            print(f" the roll is {roll}")
            initial_camera_struct['rotation_vector']=rotation_vector
            initial_camera_struct['translation_vector']=translation_vector
            initial_camera_struct['yaw']=yaw
            initial_camera_struct['pitch']=pitch
            initial_camera_struct['roll']=roll
            return initial_camera_struct, True
        prev_frame = frame

    cap.release()

process_file_for_camera_angles('images/AO.RodLaverArena1.mp4')
