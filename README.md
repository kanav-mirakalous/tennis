The following is a working model
In order to pass a video you can go to the final_pipeline.py and pass it in the process_file_for_camera_angles('images/AO.RodLaverArena1.mp4')
You can access the following elements
output_camera_struct, output_initial_camera_struct, output_mobile_struct
The output camera struct is the current camera estimates
it contains camera_struct['rotation_vector']=rotation_vector
            camera_struct['translation_vector']=translation_vector
            camera_struct['yaw']=yaw
            camera_struct['pitch']=pitch
            camera_struct['roll']=roll

The initial camera struct is the output of the same values at the end of calibration. The system essentially performs calibration waiting for upto 300 frames to detect four corners and 2 poles. once it does it outputs the initial camera struct

The output mobile struct comes from optical flow and is available to translate the points detected if needed or stop the program if too much movement is sensed
Mobile struct consists of 
mobile_struct['dx']=dx_mobile
mobile_struct['dy']=dy_mobile
mobile_struct['euler_angles']=euler_angles_mobile
