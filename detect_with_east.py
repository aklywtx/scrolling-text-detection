import os
import cv2
from mmocr.apis import MMOCRInferencer
import numpy as np
from imutils.object_detection import non_max_suppression


def east_detect(image, output_path, east_confidence):
    '''
    uses EAST detection to get x,y coordinates of all bounding boxes of input image 
    
    param images: paths to the image
    param output_path: path where to save copy of original image with bounding boxes drawn on top
    
    returns list of tuples: each tuple contains (x_min, y_min, x_max, y_max) of each detected bounding box

    code from: https://medium.com/technovators/scene-text-detection-in-python-with-east-and-craft-cbe03dda35d5
    '''
    
    
    layerNames = [
    	"feature_fusion/Conv_7/Sigmoid",
    	"feature_fusion/concat_3"]

    orig = image.copy()
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    (H, W) = image.shape[:2]
    
    # set the new width and height and then determine the ratio in change
    # for both the width and height: Should be multiple of 32
    # (newW, newH) = (320, 320)
    (newW, newH) = (1024, 1024)
    
    rW = W / float(newW)
    rH = H / float(newH)
    
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    
    (H, W) = image.shape[:2]
    
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    	(123.68, 116.78, 103.94), swapRB=True, crop=False)
    
    net.setInput(blob)
    
    (scores, geometry) = net.forward(layerNames)
    
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
    
        for x in range(0, numCols):
    		# if our score does not have sufficient probability, ignore it
            # Set minimum confidence as required
            if scoresData[x] < east_confidence:
                continue
    		# compute the offset factor as our resulting feature maps will
            #  x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
                        
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # loop over the bounding boxes
    coordinates = []

    if len(boxes) == 0: # if no text/bounding boxes detected
        return coordinates
        
    if len(boxes) >= 1:
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # draw the bounding box on the image
            # origin is upper left corner
            # startX,startY ------
            # |                 |
            # |                 |
            # |                 |
            # ----------endX,endY
            startX = max(startX, 0) #TODO: ome starting points of east text bounding boxes are negative numbers. I dont know why
            out_image = cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # out_image = cv2.rectangle(out_image, (146, 888), (1920, 1002), (0, 255, 0), 2)
            coordinates += [[startX, startY, endX, endY]]
        
        cv2.imwrite(f"{output_path}", out_image)
        return coordinates
    
    
def get_text_coordinates_in_all_frames(video_frames_dir, saved_frame_nums, east_confidence):
    """
    Get the coordinates of the bounding boxes for text detection in all frames.

    Args:
        saved_frame_nums (list): A list of frame numbers to process.
        east_confidence (float): The confidence threshold for text detection.

    Returns:
        list: A list of coordinates for each frame, where each coordinate is a tuple (x, y, w, h).

    """
    coordinates_list= []

    for frame_number in saved_frame_nums:
        frame_path = video_frames_dir + f"/frame{frame_number}.png"  # path to individual frame being processed

        base_name = os.path.basename(frame_path)
        frame_name, _ = os.path.splitext(base_name)  # get the individual frame name without the extension
        
        if not os.path.isdir(f"{video_frames_dir}/output"):
            os.makedirs(f"{video_frames_dir}/output")  # makes output directory for the frame being processed
        
        image = cv2.imread(frame_path)
        coordinates = east_detect(image, f"{video_frames_dir}/output/out_{frame_name}_EAST.png", east_confidence)  # get coordinates of bounding boxes and save image to new output dir for that frame  
        coordinates_list += [coordinates]
        
    return coordinates_list
