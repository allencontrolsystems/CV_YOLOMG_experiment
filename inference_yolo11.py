import os
from pathlib import Path
import sys
from ultralytics import YOLO
# cwd = os.getcwd().rstrip('test')
# sys.path.append(os.path.join(cwd, './'))
cwd = os.getcwd()


import cv2

import argparse
import time
from pathlib import Path
import numpy as np
from numpy import random
import json

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


def draw_predictions(img, label, score, box, color=(66, 245, 66), location=None):
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.5
    f_thickness, l_thickness = 1, 2
    
    h, w, _ = img.shape
    u1, v1, u2, v2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (u1, v1), (u2, v2), color, l_thickness)
    
    text = '%s: %.2f' % (label, score)
    text_w, text_h = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
    text_h += 6
    if v1 - text_h < 0:
        cv2.rectangle(img, (u1, text_h), (u1 + text_w, 0), color, -1)
        cv2.putText(img, text, (u1, text_h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (u1, v1), (u1 + text_w, v1 - text_h), color, -1)
        cv2.putText(img, text, (u1, v1 - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    
    if location is not None:
        text = '(%.1fm, %.1fm)' % (location[0], location[1])
        text_w, text_h = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
        text_h += 6
        if v2 + text_h > h:
            cv2.rectangle(img, (u1, h - text_h), (u1 + text_w, h), color, -1)
            cv2.putText(img, text, (u1, h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (u1, v2), (u1 + text_w, v2 + text_h), color, -1)
            cv2.putText(img, text, (u1, v2 + text_h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    
    return img

if __name__ == '__main__':
    mask = "mask31"
    demo_result_path = Path("/home/acs/YOLOMG/demo_results")
    masks_folder = demo_result_path / mask
    images_folder = demo_result_path / "images"
    inference_result_folder = demo_result_path / "inference"
    video_sets = []
    for video in images_folder.iterdir():
        video_sets.append(str(video.name))

    weights = {
        "yolo11s": "/home/acs/YOLOMG/yolo11s-pose-centerfinder-data_5frozen_best.pt"
    }


    demo_result_path = Path("/home/acs/YOLOMG/demo_results")
    images_folder = demo_result_path / "images_704"
    full_res_images_folder = demo_result_path / "images"
    inference_result_folder = demo_result_path / "inference"

    for name, weight_path in weights.items():
        print(name)
        detector = YOLO(weight_path, task="pose", verbose=False)
        for video_file in video_sets:
            print(video_file)

            inference_image_folder = inference_result_folder / name / video_file
            inference_image_resized_folder = inference_result_folder / (name + "resized_604") / video_file
            inference_image_folder.mkdir(parents=True, exist_ok=True)
            inference_image_resized_folder.mkdir(parents=True, exist_ok=True)
            video_image_folder = images_folder / video_file
            video_full_res_images_folder = full_res_images_folder / video_file

            json_file_path = video_image_folder / "crop_offset.json"
            f = open(json_file_path)
            crop_offset = json.load(f)
            f.close()

            for image_file in video_image_folder.iterdir():
                if image_file.suffix == ".jpg":
                    # for cropped image
                    full_res_image_file = video_full_res_images_folder / image_file.name
                    image = cv2.imread(str(image_file))
                    results = detector.predict(image, imgsz=704, batch=1, conf=0.1, verbose=False) # pedestrian, cyclist, car, bus, truck
                    full_res_image = cv2.imread(str(full_res_image_file))
                    img_draw = full_res_image.copy()
                    if results is not None:
                        result = results[0]
                        offsets = crop_offset[image_file.name]
                        x_offset, y_offset = offsets
                        for i in range(len(result.boxes.conf.tolist())):
                            conf = result.boxes.conf.tolist()[i]
                            x1, y1, x2, y2 = result.boxes.xyxy[i].detach().cpu().numpy().astype(int).tolist()
                            img_draw = draw_predictions(img_draw, "Drone", conf, (x1 + x_offset, y1 + y_offset, x2+x_offset, y2+y_offset))
                    image_with_border = cv2.copyMakeBorder(image, 0, img_draw.shape[0] - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                    save_image = cv2.hconcat((img_draw, image_with_border))
                    cv2.imwrite(str(inference_image_folder / image_file.name), save_image)

                    # for resized image
                    results = detector.predict(full_res_image, imgsz=704, batch=1, conf=0.1, verbose=False) # pedestrian, cyclist, car, bus, truck
                    img_draw = full_res_image.copy()
                    if results is not None:
                        result = results[0]
                        for i in range(len(result.boxes.conf.tolist())):
                            conf = result.boxes.conf.tolist()[i]
                            x1, y1, x2, y2 = result.boxes.xyxy[i].detach().cpu().numpy().astype(int).tolist()
                            img_draw = draw_predictions(img_draw, "Drone", conf, (x1, y1 , x2, y2))
                    cv2.imwrite(str(inference_image_resized_folder / image_file.name), img_draw)


