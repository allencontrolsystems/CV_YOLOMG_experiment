from pathlib import Path
import random
import json
import xml.etree.ElementTree as ET
import time
import math

import cv2
import numpy as np

from ultralytics import YOLO
from test_code.FD3_mask import FD3_mask
from dualdetector import Yolov5Detector, draw_predictions

GUNCAM_bursts = ["15_05_2025__21_16_17"]

IMAGES_PATH = Path(r"C:\Users\micha\rosie_data")
ANNOTATION_PATH = None
DESIRED_IMAGE_SAVE_PATH = Path(r"C:\Users\micha\YOLOMG\videos")

IMAGE_CROP_SIZE = 704
RANDOMNESS_FOR_CROP_IMAGE = 5
CLASSES = ["Drone"]

def tile_image(image, tile_size=(1280, 720), overlap=200):
    tiles = []
    tile_w, tile_h = tile_size

    h, w = image.shape[:2]

    step_x = tile_w - overlap
    step_y = tile_h - overlap

    num_tiles_x = math.ceil((w - overlap) / step_x)
    num_tiles_y = math.ceil((h - overlap) / step_y)

    pad_w = (num_tiles_x * step_x + overlap) - w
    pad_h = (num_tiles_y * step_y + overlap) - h

    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    for y in range(0, padded_image.shape[0] - overlap, step_y):
        for x in range(0, padded_image.shape[1] - overlap, step_x):
            tile = padded_image[y:y + tile_h, x:x + tile_w]
            tiles.append((tile, (x, y)))

    return tiles, padded_image.shape[:2]


def tile_dual_images(img1, img2=None, tile_size=(1280, 720), overlap=200):
    tiles1 = []
    tiles2 = []
    tile_w, tile_h = tile_size

    h, w = img1.shape[:2]

    step_x = tile_w - overlap
    step_y = tile_h - overlap

    num_tiles_x = math.ceil((w - overlap) / step_x)
    num_tiles_y = math.ceil((h - overlap) / step_y)

    pad_w = (num_tiles_x * step_x + overlap) - w
    pad_h = (num_tiles_y * step_y + overlap) - h

    padded_img1 = cv2.copyMakeBorder(img1, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    if img2 is not None:
        padded_img2 = cv2.copyMakeBorder(img2, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    for y in range(0, padded_img1.shape[0] - overlap, step_y):
        for x in range(0, padded_img1.shape[1] - overlap, step_x):
            tile1 = padded_img1[y:y + tile_h, x:x + tile_w]
            tiles1.append((tile1, (x, y)))
            if img2 is not None:
                tile2 = padded_img2[y:y + tile_h, x:x + tile_w]
                tiles2.append((tile2, (x, y)))

    if img2 is not None:
        return tiles1, tiles2, padded_img1.shape[:2]
    else:
        return tiles1, padded_img1.shape[:2]


def run_yolo_pose_on_tiles(tiles1, detector, conf_threshold=0.1, tiles2=None):
    detections = []

    for i in range(len(tiles1)):
        tile, (x_offset, y_offset) = tiles1[i]
        tile2, _ = tiles2[i]
        labels, scores, boxes = detector.run(tile, tile2, classes=[0, 1, 2, 3, 4])

        for box, score, label in zip(boxes, scores, labels):
            adjusted_box = [box[0] + x_offset, box[1] + y_offset, box[2] + x_offset, box[3] + y_offset]
            detections.append((adjusted_box, score))

    return detections


def visualize_detections(image, detections, tile_size=(1280, 720), overlap=200):
    tile_w, tile_h = tile_size
    step_x = tile_w - overlap
    step_y = tile_h - overlap

    h, w = image.shape[:2]

    for y in range(0, h - overlap, step_y):
        for x in range(0, w - overlap, step_x):
            cv2.rectangle(image, (x, y), (x + tile_w, y + tile_h), (200, 200, 200), 1)

    for (xmin, ymin, xmax, ymax), conf in detections:
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(image, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 2)
    return image


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
    # detector = YOLO(r"C:\Users\micha\YOLOMG\epoch19.pt")
    VIDEO_SAVE_SIZE = (1878, 1629)

    for guncam_burst in GUNCAM_bursts:
        guncam_image_path = IMAGES_PATH / guncam_burst
        print(guncam_image_path)

        image_save_path = DESIRED_IMAGE_SAVE_PATH / guncam_burst / f"rgb_images_{detector_imgsz}"
        motion_map_save_path = DESIRED_IMAGE_SAVE_PATH / guncam_burst / f"motion31_images_{detector_imgsz}"
        inference_save_path = DESIRED_IMAGE_SAVE_PATH / guncam_burst / f"inference_result_{detector_imgsz}_yolomg_tiled"
        video_save_path = DESIRED_IMAGE_SAVE_PATH / guncam_burst

        image_save_path.mkdir(parents=True, exist_ok=True)
        motion_map_save_path.mkdir(parents=True, exist_ok=True)
        inference_save_path.mkdir(parents=True, exist_ok=True)
        video_save_path.mkdir(parents=True, exist_ok=True)

        previous_1_frame = None
        previous_2_frame = None
        frame_count = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cv2_video_writer = cv2.VideoWriter(str(video_save_path / f"{guncam_burst}_{detector_imgsz}_yolomg_tiled.mp4"), fourcc, 30, VIDEO_SAVE_SIZE)
        # cv2_video_writer_motion = cv2.VideoWriter(str(video_save_path / f"{guncam_burst}_motion_{detector_imgsz}.mp4"), fourcc, 30, VIDEO_SAVE_SIZE)

        for image_file in sorted(guncam_image_path.iterdir()):

            if image_file.suffix == ".bmp":
                frame = cv2.imread(str(image_file))
                current_frame = frame
                frame_count += 1

                # tiles, padded_shape = tile_image(current_frame)
                # detections = run_yolo_pose_on_tiles(tiles, detector)
                # padded_img = cv2.copyMakeBorder(current_frame, 0, padded_shape[0] - current_frame.shape[0], 0, padded_shape[1] - current_frame.shape[1], cv2.BORDER_CONSTANT, value=0)
                # result_img = visualize_detections(padded_img, detections)
                # image_draw = visualize_detections(current_frame, detections)


                file_name_to_save = guncam_burst + '_' + str(frame_count).zfill(4)
                # cv2.imwrite(str(image_save_path / (file_name_to_save + '.jpg')), current_frame)
                if previous_2_frame is None:
                    if previous_1_frame is None:
                        previous_1_frame = current_frame
                    else:
                        previous_2_frame = current_frame
                    continue

                difference_frame = FD3_mask(previous_1_frame, previous_2_frame, current_frame).astype(np.uint8)
                difference_frame = cv2.cvtColor(difference_frame, cv2.COLOR_GRAY2BGR)
                # cv2.imwrite(str(motion_map_save_path / (guncam_burst + '_' + str(frame_count-1).zfill(4)+ '.jpg')), difference_frame)

                tiles1, tiles2, padded_shape = tile_dual_images(current_frame, difference_frame)
                detections = run_yolo_pose_on_tiles(tiles1, detector, tiles2=tiles2)

                labels, scores, boxes = detector.run(previous_2_frame, difference_frame, classes=[0, 1, 2, 3, 4])  # pedestrian, cyclist, car, bus, truck
                # image_draw = previous_2_frame.copy()
                # if labels:
                #     for i in range(len(labels)):
                #         image = draw_predictions(image_draw, labels[i], scores[i], boxes[i])
                # cv2.imwrite(str(inference_save_path / (guncam_burst + '_' + str(frame_count-1).zfill(4)+ '.jpg')), image_draw)
                padded_img1 = cv2.copyMakeBorder(current_frame, 0, padded_shape[0] - current_frame.shape[0], 0, padded_shape[1] - current_frame.shape[1], cv2.BORDER_CONSTANT, value=0)
                image_draw = visualize_detections(padded_img1, detections)
                cv2.imwrite(str(inference_save_path / (guncam_burst + '_' + str(frame_count-1).zfill(4)+ '.jpg')), image_draw)
                image_draw = cv2.resize(image_draw, VIDEO_SAVE_SIZE)
                cv2_video_writer.write(image_draw)
                # cv2_video_writer_motion.write(difference_frame)

                previous_1_frame = previous_2_frame
                previous_2_frame = current_frame

        cv2_video_writer.release()
        # cv2_video_writer_motion.release()

