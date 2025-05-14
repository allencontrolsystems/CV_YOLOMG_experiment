import os
import time
from pathlib import Path

import cv2
import numpy as np

from test_code.FD3_mask import FD3_mask

ARD_test_videos = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22', 'phantom39',
        'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54', 'phantom55', 'phantom56',
        'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom64', 'phantom73', 'phantom79',
        'phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom102', 'phantom110',
        'phantom113', 'phantom119', 'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144']

VIDEOS_PATH = Path("/home/acs/YOLOMG/full_data/ARD100_dataset/test_video")
ANNOTATION_PATH = Path("/home/acs/YOLOMG/full_data/phantom-dataset/annotations")
DESIRED_IMAGE_SAVE_PATH = Path("/home/acs/YOLOMG/evaluation_data")

IMAGE_CROP_SIZE = 704
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

if __name__ == "__main__":

    for video_count, video_name in enumerate(ARD_test_videos):

        image_save_path = DESIRED_IMAGE_SAVE_PATH / video_name / "rgb_images"
        motion_map_save_path = DESIRED_IMAGE_SAVE_PATH / video_name / "motion31_images"
        cropped_image_save_path = DESIRED_IMAGE_SAVE_PATH / video_name / "cropped_rgb_images"
        cropped_motion_map_save_path = DESIRED_IMAGE_SAVE_PATH / video_name / "cropped_motion31_images"

        image_save_path.mkdir(parents=True, exist_ok=True)
        motion_map_save_path.mkdir(parents=True, exist_ok=True)
        cropped_image_save_path.mkdir(parents=True, exist_ok=True)
        cropped_motion_map_save_path.mkdir(parents=True, exist_ok=True)

        video_path = VIDEOS_PATH / (video_name + '.mp4')

        cv2_video_capture = cv2.VideoCapture(video_path)
        previous_1_frame = None
        previous_2_frame = None
        frame_count = 0

        while cv2_video_capture.isOpened():
            ret, frame = cv2_video_capture.read()
            if not ret:
                break

            current_frame = frame
            frame_count += 1

            cv2.imwrite(str(image_save_path / (video_name + '_' + str(frame_count).zfill(4) + '.jpg')), current_frame)


            if previous_1_frame is None:
                if previous_2_frame is None:
                    previous_1_frame = current_frame
                else:
                    previous_2_frame = current_frame
                continue



