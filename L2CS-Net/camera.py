import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import time
import pathlib
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn

from l2cs import select_device, Pipeline, render

from face_detection import RetinaFace

# === Setup Gaze Estimation Pipeline ===
CWD = pathlib.Path.cwd()

gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=select_device("cuda:0" if torch.cuda.is_available() else "cpu", batch_size=1)
)

cudnn.enabled = True
bridge = CvBridge()

def callback(msg):
    try:
        # Convert ROS image to OpenCV frame
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Run gaze estimation
        start_fps = time.time()
        results = gaze_pipeline.step(frame)
        frame = render(frame, results)

        # FPS overlay
        fps = 1.0 / (time.time() - start_fps)
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Gaze Estimation", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            rospy.signal_shutdown("ESC pressed")

    except Exception as e:
        rospy.logerr(f"Gaze estimation error: {e}")

if __name__ == '__main__':
    rospy.init_node("gaze_estimation_node", anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, callback)
    rospy.loginfo("Running L2CS Gaze Estimation on /camera/color/image_raw")
    rospy.spin()
    cv2.destroyAllWindows()

