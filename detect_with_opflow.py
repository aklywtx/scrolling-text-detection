import cv2
import numpy as np

def detect_screengrab_boundingboxes(video_path):
    ''' 
    Detects significant leftward motion in the input video and returns bounding boxes of significant motion.
    
    Args:
        video_path (str): Path to the video file.

    Returns:
        list: Coordinates of bounding boxes of significant leftward motion.
    '''
    vidname = vidname = str(video_path)
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames and calculate frame at which point to save
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    save_frame = total_frames // 1.5  # save at ~75%

    ret, first_frame = cap.read() # Get the first frame: if not readable ret returns False
    if not ret:
        print("Failed to read the video.")
        return

    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Initialize a mask for consistent leftward movement
    consistent_motion_mask = None
    consistency_threshold = 0.8  # Consistent movement threshold (percentage of frames)
    num_frames = 0

    saved_screengrab = False
    screengrab_frame = None
    bounding_boxes = []

    while cap.isOpened(): # Iterate through each frame of the video
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow between current and previous frames using Farneback method
        # "flow" is a 3d array dimensions (img height, img width, 2)
        # Channel 1: Horizontal Movement. Channel 2: Vertical Movement
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 5, 15, 3, 5, 1.2, 0)
        
        # Calculate the mask of leftward movement (negative horizontal movement)
        leftward_motion_mask = flow[..., 0] < 0
        
        # Initialize the consistent motion mask
        # with the same shape as leftward_motion_mask to accumulate counts of leftward motion
        if consistent_motion_mask is None:
            consistent_motion_mask = np.zeros_like(leftward_motion_mask, dtype=np.float32)
        
        # Update the consistent leftward movement mask where leftward motion is true
        consistent_motion_mask[leftward_motion_mask] += 1
        num_frames += 1

        # Calculate the ratio of consistent leftward movement
        consistent_ratio = consistent_motion_mask / num_frames
        consistent_leftward_regions = consistent_ratio > consistency_threshold

        # Apply morphological closing to fill in the gaps
        consistent_leftward_mask = consistent_leftward_regions.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8) 
        consistent_leftward_mask = cv2.morphologyEx(consistent_leftward_mask, cv2.MORPH_CLOSE, kernel)

        # Create an image for highlighting the image where there is consistent leftward motion
        consistent_leftward_img = np.zeros_like(frame)
        consistent_leftward_img[consistent_leftward_mask == 1] = [0, 255, 0]  # Green

        # Use morphological closing (to fill small gaps) and closing (to remove noise) operations to clean leftward-motion mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(consistent_leftward_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # Save screengrab at save_frame point and extract the bounding boxes
        if num_frames == save_frame and not saved_screengrab:
            # Convert original frame to grayscale
            gray_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Overlay green highlight on grayscale frame
            overlayed_img = cv2.addWeighted(gray_frame, 0.7, consistent_leftward_img, 0.3, 0)
            screengrab_frame = overlayed_img.copy()
            saved_screengrab = True

            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Only add the bounding box if the width is at least 1.7x the height
                if w >= 1.7 * h:
                    bounding_boxes.append((x, y, x+w, y+h))
                    # Draw bounding box on the screengrab
                    cv2.rectangle(screengrab_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green bounding box

            cv2.imwrite(f'./{vidname}_75percent_80threshold.png', screengrab_frame)
    
            return bounding_boxes
    
        prev_gray = gray.copy()
    
    cap.release()
    cv2.destroyAllWindows()

    return "fail"