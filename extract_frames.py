import os
import cv2

def files(vidpath):
    """
    Extracts frames from the given video path at a specified interval.
    
    Args:
        vidpath (str): Path to the video file.

    Returns:
        tuple: Source video capture object, saved frame numbers, and directory of video frames.
    """
    base_name = os.path.basename(vidpath)
    vidname, _ = os.path.splitext(base_name)  # Split the base name to get the name without the extension
    
    try: # Remove image frames path if it already exists
        os.remove(f"{vidname}_image_frames")
    except OSError:
        pass
    if not os.path.exists(f"{vidname}_image_frames"): # create the directory if it does not already exist
        os.makedirs(f"{vidname}_image_frames")

    src_vid = cv2.VideoCapture(vidpath) # open the video file
    return(src_vid)
    
def process(vidpath, src_vid, frame_interval):
    """
    Processes video frames, saving every nth frame as specified by the interval.
    
    Args:
        src_vid (cv2.VideoCapture): Source video capture object.
        frame_interval (int): Interval for frame extraction, a hyperparameter

    Returns:
        tuple: List of saved frame numbers and the directory of video frames.
    """
    base_name = os.path.basename(vidpath)
    vidname, _ = os.path.splitext(base_name)  # Split the base name to get the name without the extension
    saved_frame_nums = []
    
    index = 0
    while src_vid.isOpened():
        ret, frame = src_vid.read() 
        if not ret: # break at the end of the video (ret returns True if frame is succesfully read)
            break
        name = f'./{vidname}_image_frames/frame{str(index)}.png'

        if index % frame_interval == 0: # every {frame_interval} frame will be saved: can adjust this number to capture more or less frames
            print('Extracting frame...' + name)
            cv2.imwrite(name, frame)
            saved_frame_nums.append(index)
        index = index + 1
        
    src_vid.release()
    cv2.destroyAllWindows()
    
    video_frames_dir = f"./{vidname}_image_frames"
    
    return saved_frame_nums, video_frames_dir