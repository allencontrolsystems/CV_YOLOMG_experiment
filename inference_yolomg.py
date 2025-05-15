import os
from pathlib import Path
import sys
# cwd = os.getcwd().rstrip('test')
# sys.path.append(os.path.join(cwd, './'))
cwd = os.getcwd()


import cv2

import argparse
import time
from pathlib import Path
import numpy as np
from numpy import random

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


def draw_predictions(img, label, score, box, color=(156, 39, 176), location=None):
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

    
class Yolov5Detector():
    def __init__(self, weights='', imgsz=1280):
        self.device = device = select_device('0')
        self.half = half = device.type != 'cpu' # half precision only supported on CUDA
        
        # Load model
        weights = os.path.join(cwd, weights) # currently just loads a single model
        self.model = model = attempt_load(weights, map_location=device) # load FP32 model
        self.stride = stride = int(model.stride.max()) # model stride
        self.imgsz = imgsz = check_img_size(imgsz, s=stride) # check img_size
        if half:
            model.half() # to FP16
        
        # Get names
        self.names = model.module.names if hasattr(model, 'module') else model.names
        
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())),torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run once

    def imgdeal(self,img):
        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float() # uint8 to fp16/32
        img /= 255.0 # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def run(self, img1, img2, conf_thres=0.1, iou_thres=0.4, classes=None, save_path1=None, save_path2=None):
        # Padded resize
        img1 = self.imgdeal(img1)
        img2 = self.imgdeal(img2)
        # print(img1.shape)
        # Inference
        pred = self.model(img1, img2, augment=False, save_path1=save_path1, save_path2=save_path2)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)
        
        # Process detections
        det = pred[0]

        if len(det):
            # Rescale boxes from imgsz to img0 size
            # boxes = scale_coords(img1.shape[2:], det[:, :4], img1.shape).round().cpu().numpy() # xyxy
            img0_shape = torch.from_numpy(np.array([1080, 1920])).to(self.device)
            boxes = scale_coords(img1.shape[2:], det[:, :4], img0_shape).round().cpu().numpy() # xyxy
            labels = [self.names[int(cls)] for cls in det[:, -1]]
            scores = [float('%.2f' % conf) for conf in det[:, -2]]
            return labels, scores, boxes
        else:
            return [], [], np.array([])


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
        # "fix0": "/home/acs/YOLOMG/runs/train/ARD_mask31_640_fix0/weights/best.pt",
        # "fix1": "/home/acs/YOLOMG/runs/train/ARD_mask31_640_fix1/weights/best.pt",
        # "fix2": "/home/acs/YOLOMG/runs/train/ARD_mask31_640_fix2/weights/best.pt",
        "fix3": "/home/acs/YOLOMG/runs/train/ARD_mask31_640_fix3/weights/best.pt"
    }


    demo_result_path = Path("/home/acs/YOLOMG/demo_results")
    masks_folder = demo_result_path / mask
    images_folder = demo_result_path / "images"
    inference_result_folder = demo_result_path / "inference"

    for name, weight_path in weights.items():
        print(name)
        detector = Yolov5Detector(weights=weight_path, imgsz=640)
        for video_file in video_sets:
            print(video_file)

            inference_image_folder = inference_result_folder / name / video_file
            inference_image_folder_sa_before = inference_result_folder / name /(video_file + "spatial_attention_before")
            inference_image_folder_sa_after = inference_result_folder / name /(video_file + "spatial_attention_after")

            inference_image_folder.mkdir(parents=True, exist_ok=True)
            inference_image_folder_sa_before.mkdir(parents=True, exist_ok=True)
            inference_image_folder_sa_after.mkdir(parents=True, exist_ok=True)

            video_mask_folder = masks_folder / video_file
            video_image_folder = images_folder / video_file

            for mask_file in video_mask_folder.iterdir():
                image_file = video_image_folder / mask_file.name
                if image_file.is_file():
                    image = cv2.imread(str(image_file))
                    mask = cv2.imread(str(mask_file))
                    save_path1 = inference_image_folder_sa_before / mask_file.name
                    save_path2 = inference_image_folder_sa_after / mask_file.name
                    labels, scores, boxes = detector.run(image, mask, classes=[0, 1, 2, 3, 4], save_path1=save_path1, save_path2=save_path2) # pedestrian, cyclist, car, bus, truck
                    if labels:
                        for i in range(len(labels)):
                            image = draw_predictions(image, labels[i], scores[i], boxes[i])
                    save_image = cv2.hconcat((image, mask))
                    cv2.imwrite(str(inference_image_folder / mask_file.name), save_image)
