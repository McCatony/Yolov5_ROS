#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

# (Optional)
import cv2
from cv_bridge import CvBridge 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes
)
from utils.plots import Annotator
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class Yolov5Detector:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold", 0.7)
        self.iou_thres = rospy.get_param("~iou_threshold", 0.45)
        self.agnostic_nms = rospy.get_param("~agnostic_nms", True)
        self.max_det = rospy.get_param("~maximum_detections", 1000)
        self.classes = rospy.get_param("~classes", None)
        
        # Initialize weights 
        weights = rospy.get_param("~weights", "yolov5s.pt")
        
        # Initialize model
        self.device = select_device(str(rospy.get_param("~device","cpu")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn", True), data=rospy.get_param("~data", ""))
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )
        
        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h", 480)]
        self.img_size = check_img_size(self.img_size, s=self.stride)
        
        # Half
        self.half = rospy.get_param("~half", False)
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.img_size))  # warmup        
        
        # Initialize subscriber from Image topic
        rospy.init_node("yolov5_ros") # Node Name
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback) # Subsribe
        self.yolo_pub = rospy.Publisher('/object', String, queue_size=10) # Publish object type
        
        # (Optional)
        self.bridge = CvBridge()
    
    def callback(self, img_msg):
        dtype = np.dtype("uint8")
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        img_np = np.ndarray(shape=(img_msg.height, img_msg.width, 3), dtype=dtype, buffer=img_msg.data)
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            img_np = img_np.byteswap().newbyteorder()
        img, img_original = self.preprocess(img_np)
        
        img_torch = torch.from_numpy(img).to(self.device) # Convert format
        img_torch = img_torch.half() if self.half else img_torch.float() # unit8 to fp16/32
        img_torch /= 255
        if len(img_torch.shape) == 3:
            # Add an extra demension at the beginning for torch
            img_torch = img_torch[None]
        
        # Execute object detection and Collect grids
        pred = self.model(img_torch, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        
        det = pred[0].cpu().numpy()
        
        # Publish object type
        if len(det):
            # Rescale boxes from img size to img_original size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_original.shape).round()
            for *xyxy, conf, cls in reversed(det) :
                c = int(cls)
                
                if c : 
                    detected_object = self.names[c]
                    
                    # Publish object type
                    self.yolo_pub.publish(detected_object)
        
        # (Optional) Observe what is captured
        img_cv = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        cv2.imshow(str(0), img_cv)
        cv2.waitKey(1)
    
    def preprocess(self, img):
        img_original = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, B(atch)H(eight)W(idth)C(hannel) to BCHW
        img = np.ascontiguousarray(img)
        
        return img, img_original

def main() :     
    object_detection = Yolov5Detector()
    
    rospy.spin()

if __name__ == "__main__":
    main()
