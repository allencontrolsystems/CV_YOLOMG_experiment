from typing import Tuple, List
from enum import Enum
from pathlib import Path
import os
import sys

import numpy as np
import torch
import ultralytics

sys.path.append(os.path.abspath('../cv-common'))
from inference.CenterFinderBase import CenterFinderBase, DroneCenterPixel, BoundingBox, DEVICE
from evaluation.static_image_evaluation import StaticImageEvaluation

class CenterFinderYolo(CenterFinderBase):

    def __init__(self,
                 model_weight: str = None,
                 conf: float = 0.1,
                 verbose: bool = False
                 ):

        self.verbose = verbose

        self.model = ultralytics.YOLO(
            model_weight,
            task="pose",
            verbose=verbose)
        self.conf = conf

        self.model.to("cuda:1")

        if DEVICE == torch.device("cpu"):
            print("\n", "GPU NOT DETECTED, WILL INFERENCE ON CPU", "\n")

    def get_centers(self,
                    img: np.array,
                    iou_threshold: float = 0.7,
                    max_det: int = 100,
                    img_sz: Tuple[int, int] = None,
                    **kwargs) -> List[DroneCenterPixel]:
        """Returns the centers of the detected object in the image(sorted by highest confidence first), if
        an object was detected. If no object is detected, returns empty list.

        Args:
            img: An input image as a numpy array. Expects
                the array to be an RGB image of shape
                (height, width, channels).
            conf: Sets the minimum confidence threshold for detections.
                  Objects detected with confidence below this threshold will be disregarded
            iou_threshold: Discards all overlapping bboxes with IoU > iou_threshold during inference
            max_det: Maximum number of detections allowed per image.
            img_sz: Inference image size, defaults to None which tries to use the entire image.

        Returns:
            List[CenterXY]
        """

        img_sz = img_sz if img_sz else img.shape[:2]  # Take just the (H, W)
        results = self.model.predict(img, imgsz=img_sz, batch=1, verbose=self.verbose,
                                     conf=self.conf, iou=iou_threshold, max_det=max_det)
        centers = []

        if results is not None:
            result = results[0]  # get the results corresponding to first batch

            for i in range(len(result.boxes.conf.tolist())):
                conf = result.boxes.conf.tolist()[i]
                x1, y1, x2, y2 = result.boxes.xyxy[i].detach().cpu().numpy().astype(int).tolist()
                raw_center = result.keypoints.xy[i].detach().cpu().squeeze().tolist()

                current_bounding_box = BoundingBox(start_x=x1, start_y=y1,
                                                   end_x=x2, end_y=y2)
                new_center = DroneCenterPixel(x=float(raw_center[0]),
                                              y=float(raw_center[1]),
                                              confidence=conf,
                                              bounding_box=current_bounding_box)
                centers.append(new_center)

            centers = sorted(centers, key=lambda x: x.confidence, reverse=True)

        return centers

if __name__ == "__main__":

    model_path = "epoch19.pt"
    conf_threshold = 0.01

    center_finder = CenterFinderYolo(model_weight=model_path, conf=conf_threshold)
    static_image_evaluation = StaticImageEvaluation(center_finder, "evaluation.json")
    static_image_evaluation.evaluate()