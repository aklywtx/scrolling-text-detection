import cv2
from PIL import Image
from mmocr.apis import MMOCRInferencer

# crop and save bounding boxes
def crop_and_save_image(image_path, coords, output_path):
    '''Saves crops of the original image according to specified coordinates.

    Args:
        image_path (str): Path to the original image.
        coords (tuple): Coordinates of the bounding box in the format (x_min, y_min, x_max, y_max).
        output_path (str): Path of the output file.

    Returns:
        None

    Raises:
        cv2.error: If there is an error while saving the cropped image.

    '''
    
    img = cv2.imread(image_path)
    x_min, y_min, x_max, y_max = coords
    cropped_img = img[y_min:y_max, x_min:x_max]
    
    # Save
    try:
        cv2.imwrite(output_path, cropped_img)
    except cv2.error as e:
        print(coords)
        print(f"OpenCV error: {e}")

# adding borders to the boxes:
def add_borders(input_img_path, output_img_path):
	"""
	Add borders to the input image.

	Parameters:
	input_img_path (str): The file path of the input image.
	output_img_path (str): The file path to save the output image.

	Returns:
	None
	"""
	old_im = Image.open(input_img_path)
	old_size = old_im.size
	
	new_size = (old_size[0] + 320, old_size[1] + 512)
	new_im = Image.new("RGB", new_size, (255, 255, 255))   ## luckily, this is already black!
	box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
	new_im.paste(old_im, box)
	
	new_im.save(output_img_path)
    
# Load models into memory
def get_ocr_texts(output_path):
	"""
	Extracts and sorts the OCR texts from the given output path.

	Args:
		output_path (str): The path to the OCR output file.

	Returns:
		list: A list of sorted OCR texts.
	"""
	ocr = MMOCRInferencer(det='DBNet', rec='SATRN', device='cpu')

	# Perform inference
	results = ocr(output_path, show=False, print_result=False)

	# Extract the OCR texts as a list of tuples with (word bounding box first coorinate, word)
	tuples_list = [(polygon[0], text) for polygon, text in zip(results["predictions"][0]["det_polygons"], results["predictions"][0]["rec_texts"])]

	# Sort the list of tuples based on the first number(word bounding box first coorinate) in each tuple
	sorted_tuples = sorted(tuples_list, key=lambda x: x[0])

	# Extract the sorted texts
	sorted_ocr_texts = [text for _, text in sorted_tuples]
 
	return sorted_ocr_texts