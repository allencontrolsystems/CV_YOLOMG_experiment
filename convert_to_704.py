import xml.etree.ElementTree as ET
from pathlib import Path
import random
import json

import cv2

CLASSES = ["Drone"]

def get_boundingbox(xml_file):
    in_file = open(xml_file, encoding='UTF-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):

        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b

        if b1 == 0:
            b1 = 1
        if b3 == 0:
            b3 = 1
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        return (b1, b2, b3, b4)
    
def crop_fixed_size(image_shape, x_min, x_max, y_min, y_max, image_crop_size, randomness=0.25):

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    max_offset_x = int((x_max - x_min) * randomness)
    max_offset_y = int((y_max - y_min) * randomness)

    offset_x = random.randint(-max_offset_x, max_offset_x)
    offset_y = random.randint(-max_offset_y, max_offset_y)

    random_x_center = x_center + offset_x
    random_y_center = y_center + offset_y

    # Compute crop boundaries
    x_start = int(max(0, random_x_center - image_crop_size / 2))
    y_start = int(max(0, random_y_center - image_crop_size / 2))

    x_end = min(image_shape[1], x_start + image_crop_size)
    y_end = min(image_shape[0], y_start + image_crop_size)

    if x_end - x_start < image_crop_size:
        x_start = max(0, x_end - image_crop_size)
    if y_end - y_start < image_crop_size:
        y_start = max(0, y_end - image_crop_size)
    
    return int(x_start), int(x_end), int(y_start), int(y_end)

if __name__ == '__main__':
    IMAGE_CROP_SIZE = 704
    RANDOMNESS = 5
    mask = "mask31"
    annotation_folder = Path("/home/acs/YOLOMG/full_data/phantom-dataset/annotations")
    demo_result_path = Path("/home/acs/YOLOMG/demo_results")
    masks_folder = demo_result_path / mask
    images_folder = demo_result_path / "images"

    video_sets = []
    for video in images_folder.iterdir():
        video_sets.append(str(video.name))

    for video_file in video_sets:
        print(video_file)
        cropped_image_folder = demo_result_path / (f"images_{IMAGE_CROP_SIZE}") / video_file
        cropped_image_folder.mkdir(parents=True, exist_ok=True)
        video_mask_folder = masks_folder / video_file
        video_image_folder = images_folder / video_file
        video_annotation_folder = annotation_folder / video_file 

        crop_offset = {}

        for mask_file in video_mask_folder.iterdir():
            annotation_file = video_annotation_folder / (mask_file.stem + ".xml")
            image_file = video_image_folder / mask_file.name
            cropped_image_file = cropped_image_folder / mask_file.name

            bgr_image = cv2.imread(str(image_file))
            bbox = get_boundingbox(annotation_file)
            if bbox is not None:
                x_min, x_max, y_min, y_max = bbox
                x_min, x_max, y_min, y_max = crop_fixed_size(bgr_image.shape, x_min, x_max,y_min, y_max, IMAGE_CROP_SIZE, RANDOMNESS)

                cropped_bgr_image = bgr_image[y_min:y_max, x_min:x_max, :]
                crop_offset[mask_file.name] = [x_min, y_min]
                cv2.imwrite(str(cropped_image_file), cropped_bgr_image)
        
        crop_offset_file = cropped_image_folder / "crop_offset.json"
        with open(crop_offset_file, 'w') as f:
            json.dump(crop_offset, f)

