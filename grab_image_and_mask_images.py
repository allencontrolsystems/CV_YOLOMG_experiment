from pathlib import Path
import random
import json
import xml.etree.ElementTree as ET
import time

import cv2
import numpy as np

from test_code.FD3_mask import FD3_mask
from dualdetector import Yolov5Detector, draw_predictions

ARD_test_videos = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22', 'phantom39',
        'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54', 'phantom55', 'phantom56',
        'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom64', 'phantom73', 'phantom79',
        'phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom102', 'phantom110',
        'phantom113', 'phantom119', 'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144']

GUNCAM_bursts = ["15_05_2025__22_03_51"]

IMAGES_PATH = Path(r"C:\Users\micha\Downloads\from_google_drive")
ANNOTATION_PATH = None
DESIRED_IMAGE_SAVE_PATH = Path(r"C:\Users\micha\YOLOMG\videos")

IMAGE_CROP_SIZE = 704
RANDOMNESS_FOR_CROP_IMAGE = 5
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
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        xmin, xmax, ymin, ymax = b

        if xmin == 0:
            xmin = 1
        if ymin == 0:
            ymin = 1
        if xmax > w:
            xmax = w
        if ymax > h:
            ymax = h

        return (xmin, xmax, ymin, ymax)
    return None


def get_random_crop_around_target(image_shape, x_min, x_max, y_min, y_max, image_crop_size, randomness=0.25):
    # Calculate center of bounding box
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    max_offset_x = int((x_max - x_min) * randomness)
    max_offset_y = int((y_max - y_min) * randomness)

    offset_x = random.randint(-max_offset_x, max_offset_x)
    offset_y = random.randint(-max_offset_y, max_offset_y)

    random_x_center = x_center + offset_x
    random_y_center = y_center + offset_y

    x_start = max(0, random_x_center - image_crop_size // 2)
    y_start = max(0, random_y_center - image_crop_size // 2)

    x_end = min(image_shape[1], x_start + image_crop_size)
    y_end = min(image_shape[0], y_start + image_crop_size)

    # Adjust crop start if we hit the edge
    if x_end - x_start < image_crop_size:
        x_start = max(0, x_end - image_crop_size)
    if y_end - y_start < image_crop_size:
        y_start = max(0, y_end - image_crop_size)

    # Adjust bounding box coordinates for cropped image
    new_x_min = max(0, x_min - x_start)
    new_x_max = min(image_crop_size, x_max - x_start)
    new_y_min = max(0, y_min - y_start)
    new_y_max = min(image_crop_size, y_max - y_start)

    return (int(x_start), int(x_end), int(y_start), int(y_end)), (new_x_min, new_x_max, new_y_min, new_y_max)

def save_annotation(save_path, x_min=None, x_max=None, y_min=None, y_max=None):

    annotation = {}
    if x_min is not None:
        annotation["bbox"] = [[x_min, y_min, x_max, y_max]]
    else:
        annotation["bbox"] = [[]]

    with open(save_path, 'w') as f:
        json.dump(annotation, f)


if __name__ == "__main__":

    detector_imgsz = 1280
    detector = Yolov5Detector(r"C:\Users\micha\YOLOMG\runs\train\ARD100_mask32-1280_uavs\weights\best.pt", imgsz=detector_imgsz)
    VIDEO_SAVE_SIZE = (3756, 3258)

    for guncam_burst in GUNCAM_bursts:
        guncam_image_path = IMAGES_PATH / guncam_burst

        image_save_path = DESIRED_IMAGE_SAVE_PATH / guncam_burst / "rgb_images"
        motion_map_save_path = DESIRED_IMAGE_SAVE_PATH / guncam_burst / "motion31_images"
        inference_save_path = DESIRED_IMAGE_SAVE_PATH / guncam_burst / "inference_result"
        video_save_path = DESIRED_IMAGE_SAVE_PATH / guncam_burst

        image_save_path.mkdir(parents=True, exist_ok=True)
        motion_map_save_path.mkdir(parents=True, exist_ok=True)
        inference_save_path.mkdir(parents=True, exist_ok=True)
        video_save_path.mkdir(parents=True, exist_ok=True)

        previous_1_frame = None
        previous_2_frame = None
        frame_count = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cv2_video_writer = cv2.VideoWriter(str(video_save_path / f"{guncam_burst}_{detector_imgsz}.mp4"), fourcc, 30, VIDEO_SAVE_SIZE)

        for image_file in sorted(guncam_image_path.iterdir()):

            if image_file.suffix == ".bmp":
                frame = cv2.imread(str(image_file))
                current_frame = frame
                frame_count += 1

                file_name_to_save = guncam_burst + '_' + str(frame_count).zfill(4)
                cv2.imwrite(str(image_save_path / (file_name_to_save + '.jpg')), current_frame)
                if previous_2_frame is None:
                    if previous_1_frame is None:
                        previous_1_frame = current_frame
                    else:
                        previous_2_frame = current_frame
                    continue

                difference_frame = FD3_mask(previous_1_frame, previous_2_frame, current_frame).astype(np.uint8)
                difference_frame = cv2.cvtColor(difference_frame, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(motion_map_save_path / (guncam_burst + '_' + str(frame_count-1).zfill(4)+ '.jpg')), difference_frame)

                labels, scores, boxes = detector.run(previous_2_frame, difference_frame, classes=[0, 1, 2, 3, 4])  # pedestrian, cyclist, car, bus, truck
                image_draw = previous_2_frame.copy()
                if labels:
                    for i in range(len(labels)):
                        image = draw_predictions(image_draw, labels[i], scores[i], boxes[i])
                cv2.imwrite(str(inference_save_path / (guncam_burst + '_' + str(frame_count-1).zfill(4)+ '.jpg')), image_draw)

                image_draw = cv2.resize(image_draw, VIDEO_SAVE_SIZE)
                cv2_video_writer.write(image_draw)

                previous_1_frame = previous_2_frame
                previous_2_frame = current_frame

        cv2_video_writer.release()

