from pathlib import Path
import random
import json
import xml.etree.ElementTree as ET
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2

from test_code.FD3_mask import FD3_mask

ARD_test_videos = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22', 'phantom39',
        'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54', 'phantom55', 'phantom56',
        'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom64', 'phantom73', 'phantom79',
        'phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom102', 'phantom110',
        'phantom113', 'phantom119', 'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144']

ARD_train_videos = ['phantom09', 'phantom10', 'phantom14', 'phantom17', 'phantom19', 'phantom20', 'phantom28', 'phantom29', 'phantom30', 'phantom32',
        'phantom36', 'phantom40', 'phantom42', 'phantom43', 'phantom44', 'phantom46', 'phantom63', 'phantom65', 'phantom66', 'phantom68',
        'phantom70', 'phantom71', 'phantom74', 'phantom75', 'phantom76', 'phantom77', 'phantom78', 'phantom80', 'phantom81', 'phantom82',
        'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89', 'phantom90', 'phantom101', 'phantom103', 'phantom104', 'phantom105',
        'phantom106', 'phantom107', 'phantom108', 'phantom109', 'phantom111', 'phantom112', 'phantom114', 'phantom115', 'phantom116', 'phantom117',
        'phantom118', 'phantom120', 'phantom132', 'phantom137', 'phantom138', 'phantom139', 'phantom140', 'phantom142', 'phantom143', 'phantom145',
        'phantom146', 'phantom147', 'phantom148', 'phantom149', 'phantom150']

VIDEOS_PATH = Path("/home/acs/YOLOMG/full_data/ARD100_dataset/train_video")
ANNOTATION_PATH = Path("/home/acs/YOLOMG/full_data/phantom-dataset/annotations")
DESIRED_IMAGE_SAVE_PATH = Path("/home/acs/YOLOMG/cropped_training_data")
DESIRED_yolo11_IMAGE_SAVE_PATH = Path("/home/acs/YOLOMG/yolo11_cropped_training_data")
DESIRED_cropped_yolomg_PATH = Path("/home/acs/YOLOMG/full_data/ARD100_cropped/")

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

def save_normalized_annotation(save_path, x_min=None, x_max=None, y_min=None, y_max=None, save_kp=False):
    
    bbox_center_x, bbox_center_y = (x_max + x_min) / 2, (y_max + y_min) / 2
    center_x, center_y = bbox_center_x, bbox_center_y
    bbox_height, bbox_width = y_max - y_min, x_max - x_min

    normalized_center_x, normalized_center_y = center_x / IMAGE_CROP_SIZE, center_y / IMAGE_CROP_SIZE
    normalized_bbox_width, normalized_bbox_height = bbox_width / IMAGE_CROP_SIZE, bbox_height / IMAGE_CROP_SIZE
    normalized_bbox_center_x, normalized_bbox_center_y = bbox_center_x / IMAGE_CROP_SIZE, bbox_center_y / IMAGE_CROP_SIZE

    with open(save_path, "w") as f:
        if save_kp: 
            f.write(f"0 {normalized_center_x} {normalized_center_y} {normalized_bbox_width} {normalized_bbox_height} {normalized_bbox_center_x} {normalized_bbox_center_y}")
        else:
            f.write(f"0 {normalized_center_x} {normalized_center_y} {normalized_bbox_width} {normalized_bbox_height}")

def get_training_data(video_name):

        # image_save_path = DESIRED_IMAGE_SAVE_PATH / video_name / "rgb_images"
        # motion_map_save_path = DESIRED_IMAGE_SAVE_PATH / video_name / "motion31_images"
        # cropped_image_save_path = DESIRED_IMAGE_SAVE_PATH / video_name / "cropped_rgb_images"
        # cropped_motion_map_save_path = DESIRED_IMAGE_SAVE_PATH / video_name / "cropped_motion31_images"
        yolo11_cropped_image_save_path = DESIRED_yolo11_IMAGE_SAVE_PATH / "images"
        yolo11_cropped_label_save_path = DESIRED_yolo11_IMAGE_SAVE_PATH / "labels"
        yolomg_cropped_image_save_path = DESIRED_cropped_yolomg_PATH / "images"
        yolomg_cropped_label_save_path = DESIRED_cropped_yolomg_PATH / "labels"
        yolomg_cropped_image2_save_path = DESIRED_cropped_yolomg_PATH / "mask31"

        # image_save_path.mkdir(parents=True, exist_ok=True)
        # motion_map_save_path.mkdir(parents=True, exist_ok=True)
        # cropped_image_save_path.mkdir(parents=True, exist_ok=True)
        # cropped_motion_map_save_path.mkdir(parents=True, exist_ok=True)
        yolo11_cropped_image_save_path.mkdir(parents=True, exist_ok=True)
        yolo11_cropped_label_save_path.mkdir(parents=True, exist_ok=True)
        yolomg_cropped_image_save_path.mkdir(parents=True, exist_ok=True)
        yolomg_cropped_label_save_path.mkdir(parents=True, exist_ok=True)
        yolomg_cropped_image2_save_path.mkdir(parents=True, exist_ok=True)

        video_path = VIDEOS_PATH / (video_name + '.mp4')

        cv2_video_capture = cv2.VideoCapture(video_path)
        previous_1_frame = None
        previous_2_frame = None
        frame_count = 0


        previous_crop_x_start, previous_crop_x_end, previous_crop_y_start, previous_crop_y_end = 0, 0, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE
        while cv2_video_capture.isOpened():
            ret, frame = cv2_video_capture.read()
            if not ret:
                break

            current_frame = frame
            frame_count += 1

            file_name_to_save = video_name + '_' + str(frame_count).zfill(4)
            # cv2.imwrite(str(image_save_path / (file_name_to_save + '.jpg')), current_frame)
            annotation_xml_path = Path(ANNOTATION_PATH / video_name / (file_name_to_save + '.xml'))

            bbox = get_boundingbox(annotation_xml_path)
            if bbox is not None: # case there is annotation for drone
                x_min, x_max, y_min, y_max = bbox

                (crop_x_start, crop_x_end, crop_y_start, crop_y_end), (new_x_min, new_x_max, new_y_min, new_y_max) = get_random_crop_around_target(current_frame.shape, x_min, x_max, y_min, y_max, image_crop_size=IMAGE_CROP_SIZE, randomness=RANDOMNESS_FOR_CROP_IMAGE)
                cropped_current_frame = current_frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end, :]

                # cv2.imwrite(str(cropped_image_save_path / (video_name + '_' + str(frame_count).zfill(4) + '.jpg')), cropped_current_frame)

                # save_annotation(image_save_path / (file_name_to_save + ".json"), x_min, x_max, y_min, y_max)
                # save_annotation(cropped_image_save_path / (file_name_to_save + ".json"), new_x_min, new_x_max, new_y_min, new_y_max)

                # save training data for yolo11
                cv2.imwrite(str(yolo11_cropped_image_save_path / (video_name + '_' + str(frame_count).zfill(4) + '.jpg')), cropped_current_frame)
                save_normalized_annotation(str(yolo11_cropped_label_save_path / (video_name + '_' + str(frame_count).zfill(4) + '.txt')), new_x_min, new_x_max, new_y_min, new_y_max, save_kp=True)
                
                previous_crop_x_start, previous_crop_x_end, previous_crop_y_start, previous_crop_y_end = crop_x_start, crop_x_end, crop_y_start, crop_y_end

            else: # case there is no annotation for drone
                cropped_current_frame = current_frame[previous_crop_y_start:previous_crop_y_end, previous_crop_x_start:previous_crop_x_end, :]
                # cv2.imwrite(str(cropped_image_save_path / (video_name + '_' + str(frame_count).zfill(4) + '.jpg')), cropped_current_frame)
                # save_annotation(image_save_path / (file_name_to_save + ".json"))
                # save_annotation(cropped_image_save_path / (file_name_to_save + ".json"))

            if previous_2_frame is None:
                if previous_1_frame is None:
                    previous_1_frame = current_frame
                else:
                    previous_2_frame = current_frame
                continue

            difference_frame = FD3_mask(previous_1_frame, previous_2_frame, current_frame, video_name, frame_count-1)
            # cv2.imwrite(motion_map_save_path / (video_name + '_' + str(frame_count-1).zfill(4)+ '.jpg'), difference_frame)
            cropped_difference_frame = difference_frame[previous_crop_y_start:previous_crop_y_end, previous_crop_x_start:previous_crop_x_end]
            # cv2.imwrite(cropped_motion_map_save_path / (video_name + '_' + str(frame_count-1).zfill(4) + '.jpg'), cropped_difference_frame)

            cv2.imwrite(str(yolomg_cropped_image_save_path / (video_name + '_' + str(frame_count-1).zfill(4) + '.jpg')), cropped_current_frame)
            cv2.imwrite(str(yolomg_cropped_image2_save_path / (video_name + '_' + str(frame_count-1).zfill(4) + '.jpg')), cropped_difference_frame)
            if bbox is not None:
                save_normalized_annotation(str(yolomg_cropped_label_save_path / (video_name + '_' + str(frame_count-1).zfill(4) + '.txt')), new_x_min, new_x_max, new_y_min, new_y_max, save_kp=False)

            previous_1_frame = previous_2_frame
            previous_2_frame = current_frame

        cv2_video_capture.release()

if __name__ == "__main__":

    with ThreadPoolExecutor(max_workers=48) as e:
        for video_name in ARD_train_videos:
            e.submit(get_training_data, video_name)


    # for video_count, video_name in enumerate(ARD_train_videos):

        # t2 = time.time()
        # took = t2 - t1
        # videos_left_to_go = videos_len - video_count - 1
        # print(f"{videos_left_to_go} more videos to go and last video took {took} s, estimate to take {round(took * videos_left_to_go / 60, 2)} more mins")



