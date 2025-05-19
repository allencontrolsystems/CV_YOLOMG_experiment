from typing import Tuple, List
from enum import Enum
from pathlib import Path
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath('../cv-common'))
from inference.CenterFinderBase import CenterFinderBase, DroneCenterPixel, BoundingBox, DEVICE
from inference_yolomg import Yolov5Detector

class CenterFinderYoloMG(CenterFinderBase):

    def __init__(self,
                 model_weight: str = None,
                 conf: float = 0.1,
                 imgsz=1280
                 ):

        self.model = Yolov5Detector(model_weight, imgsz)
        self.conf = conf

        if DEVICE == torch.device("cpu"):
            print("\n", "GPU NOT DETECTED, WILL INFERENCE ON CPU", "\n")

    def get_centers(self,
                    rgb_img: np.array,
                    motion_img: np.array,
                    iou_threshold: float = 0.7,
                    max_det: int = 100,
                    img_sz: Tuple[int, int] = None) -> List[DroneCenterPixel]:

        labels, scores, boxes = self.model.run(rgb_img, motion_img, classes=[0, 1, 2, 3, 4])
        centers = []

        if labels:

            for i in range(len(labels)):

                conf = scores[i]
                x1, y1, x2, y2 = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2


                current_bounding_box = BoundingBox(start_x=x1, start_y=y1,
                                                   end_x=x2, end_y=y2)
                new_center = DroneCenterPixel(x=center_x,
                                              y=center_y,
                                              confidence=conf,
                                              bounding_box=current_bounding_box)
                centers.append(new_center)

            centers = sorted(centers, key=lambda x: x.confidence, reverse=True)

        return centers