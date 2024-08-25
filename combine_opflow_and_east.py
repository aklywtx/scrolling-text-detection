import numpy as np
import random


def calculate_overlap_area(rect1, rect2):
    """
    Calculate the overlapping area between two rectangles.
    
    Args:
        rect1 (tuple): The coordinates of the first rectangle in the form (startX, startY, endX, endY).
        rect2 (tuple): The coordinates of the second rectangle in the form (startX, startY, endX, endY).
    
    Returns:
        float: The overlapping area between the two rectangles.
    """
    x1_max = max(rect1[0], rect2[0])
    y1_max = max(rect1[1], rect2[1])
    x2_min = min(rect1[2], rect2[2])
    y2_min = min(rect1[3], rect2[3])
    
    overlap_width = max(0, x2_min - x1_max)
    overlap_height = max(0, y2_min - y1_max)
    
    if x1_max >= x2_min or y1_max >= y2_min:
        return 0
    
    return overlap_width * overlap_height


def calculate_total_overlapping_area(main_rect, rectangles):
    """
    Calculate the total overlapping area of multiple rectangles with the main rectangle.
    
    Args:
        main_rect (tuple): A tuple representing the main rectangle in the form (startX, startY, endX, endY).
        rectangles (list): A list of tuples representing the rectangles to calculate the overlapping areas with.
                           Each tuple should be in the form (startX, startY, endX, endY).
    
    Returns:
        float: The total overlapping area between the main rectangle and the given rectangles.
    """
    total_overlap_area = 0
    for rect in rectangles:
        total_overlap_area += calculate_overlap_area(main_rect, rect)
    
    return total_overlap_area


def get_best_opflow_box(saved_frame_nums, opflow_box_coordinates, coordinates_list, num_frames):
    """
    Calculates the rolling box with the maximum overlap area.

    Args:
        opflow_box_coordinates (list): List of opflow box coordinates.
        coordinates_list (list): List of coordinates for each frame.
        num_frames (int): Number of frames to consider for overlap check.

    Returns:
        tuple: The rolling box with the maximum overlap area.

    """
    overlap_areas = []
    chosen_frames = random.sample(list(range(len(saved_frame_nums))), num_frames)
    
    for opflow_box_coordinate in opflow_box_coordinates:
        total_overlap = 0
        for frame_index in chosen_frames:
            total_overlap += calculate_total_overlapping_area(opflow_box_coordinate, coordinates_list[frame_index])
        overlap_areas.append(total_overlap)

    rolling_box_index = np.argmax(np.array(overlap_areas))
    rolling_box = opflow_box_coordinates[rolling_box_index]
    
    return rolling_box