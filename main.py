#!/usr/bin/env python
# coding: utf-8

import os, sys
import cv2
import numpy as np
import random
import time
import argparse

from extract_frames import files, process
from detect_with_opflow import detect_screengrab_boundingboxes
from detect_with_east import get_text_coordinates_in_all_frames
from combine_opflow_and_east import get_best_opflow_box
from ocr import crop_and_save_image, add_borders, get_ocr_texts
from concat import llama3



random.seed(42)

parser = argparse.ArgumentParser()
# # hyperparameters
parser.add_argument("-i", "--frame_interval", type=int, default=50, help="interval for extracting frames")
parser.add_argument("-c", "--east_confidence", type=float, default=0.5, help="confidence threshold for EAST text detection")
parser.add_argument("-n", "--num_frames_for_overlap_check", type=int, default=3, help="number of frames for checking the overlap of text boxes and optical flow boxes")

parser.add_argument("-v", "--vidpath", type=str, default='data/train5_crop.mp4', help="video path")

args = parser.parse_args()
# print(args.language)

if __name__ == "__main__":
    # Step 1: Extract frames from video
    vid = files(args.vidpath)
    saved_frame_nums, video_frames_dir = process(args.vidpath, vid, args.frame_interval)

    # Step 2: 
    # Use Optical Flow to get rolling text bouding boxes candidates
    # Use EAST Flow to get rolling text bouding boxes candidates
    start = time.time()
    opflow_box_coordinates = detect_screengrab_boundingboxes(args.vidpath)
    print(f"{args.vidpath} time taken: {time.time()-start}")
    print(f"optical flow box coordinates: {opflow_box_coordinates}")

    coordinates_list = get_text_coordinates_in_all_frames(video_frames_dir, saved_frame_nums, args.east_confidence)
    print(f"east word box coordinates: {coordinates_list}")

    # Step 3: 
    # Combine Opflow and EAST to get the best optical flow box
    rolling_box = get_best_opflow_box(saved_frame_nums, opflow_box_coordinates, coordinates_list, args.num_frames_for_overlap_check)
    print(f"best optical flow box: {rolling_box}")


    # Step 4: Use mmocr to extract texts in rolling boxes
    # After getting the rolling box, we crop the image with opflow boxes
    # but mmocr works badly on them....
    # Adding a wide border around these boxes can give us good results!
    # (Maybe because these boxes are too flat)

    """
    1. Creates a directory to store cropped opflow images.
    2. Iterates over a list of frame numbers and performs the following operations for each frame:
    a. Retrieves the path of the frame image.
    b. Crops and saves the opflow image using a rolling box.
    c. Prints the path of the saved cropped opflow image.
    d. Adds borders to the cropped opflow image.
    e. Extracts OCR texts from the opflow image with borders.
    f. Prints the OCR texts for the current frame.
    g. Appends the OCR texts to a list.
    """
    ocr_texts = []

    output_path = os.path.join(video_frames_dir, "croppedOpflow")
    if os.path.isdir(output_path):
        print("The cropped folders exists! Please delete them :)")
    else:
        os.makedirs(output_path)

    for frame_num in saved_frame_nums:
        frame_path = os.path.join(video_frames_dir, f"frame{frame_num}.png")
            
        opflow_crop_path = os.path.join(output_path, f"frame_num_{frame_num}.jpg")
        crop_and_save_image(frame_path, rolling_box, opflow_crop_path)
        
        print(f"Saved cropped opflow box to {opflow_crop_path}")
        
        opflow_crop_with_border_path = os.path.join(output_path, f"frame_num_{frame_num}_border.jpg")
        add_borders(opflow_crop_path, opflow_crop_with_border_path)
        cur_ocr_texts = get_ocr_texts(opflow_crop_with_border_path)
        print(f"Current frame ocr texts are {cur_ocr_texts}")
        ocr_texts.append(cur_ocr_texts)


    for text in ocr_texts:
        print(text)


    # Step 5: Use llama to concatenate the texts
    prompt = f"Can you please take these word lists extracted from each frame of a rolling text video, concatenate them based on overlapping parts, and provide me with the resulting sentence? Please keep the original language, don't translate. Don't explain; provide only the sentence. {ocr_texts}"
    response = llama3(prompt)
    print(response)