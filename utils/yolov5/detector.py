
import argparse
import enum
import os
import sys
from pathlib import Path
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np


from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.augmentations import letterbox

class Detector:
    def __init__(self) -> None:
        weight_path = str(Path(os.path.dirname(__file__)).joinpath("weights_v5n/best.pt").resolve())
        self.model = DetectMultiBackend(weight_path)
        self.stride, names, self.pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(640, s=self.stride)
        print("finished setup")
        
    def detect(self, img0):
        img = letterbox(img0, self.imgsz, stride=self.stride, auto=self.pt)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to("cpu")
        img = img.float()
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        pred = self.model(img)
        conf_thres=float(0.25),  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        classes=None
        agnostic_nms=False
        max_det=1000
        pred = non_max_suppression(pred, conf_thres=0.25, classes=classes, agnostic=agnostic_nms, max_det=max_det)
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        results = []
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                exist = False
                for k, r in enumerate(results):
                    if r["cls"] == cls and r["conf"] < conf:
                        exist = True
                        results[k] = {"xyxy": torch.tensor(xyxy).tolist(), "cls": int(cls.item()), "conf": conf.item()}
                if not exist:
                    results.append({"xyxy": torch.tensor(xyxy).tolist(), "cls": int(cls.item()), "conf": conf.item()})
        return results
        


if __name__ == "__main__":
    path = str(Path(os.path.dirname(__file__)).joinpath("images/fixed_ssbu_101.png").resolve())
    img0 = cv2.imread(path)
    detector = Detector()
    t1 = time.time()
    results = detector.detect(img0)
    elapsed_time = time.time() - t1
    print("finished {}".format(elapsed_time))
    print(results) # cls 0: purple mario, 1: red mario 2: stage
