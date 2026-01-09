import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import threading
import queue
import os
import pathlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from l2cs import select_device, Pipeline, render

# === Globals ===
CWD = pathlib.Path.cwd()
HEADLESS = os.environ.get('HEADLESS', '0') == '1'
bridge = CvBridge()
frame_queue = queue.Queue(maxsize=1)
display_frame = None
lock = threading.Lock()

video_writer = None
video_output_path = str(CWD / 'output_gaze_estimation.avi')
video_size = (640, 480)  # set this correctly after reading first frame

# === Load Gaze Pipeline ===
device = select_device("gpu:0" if torch.cuda.is_available() else "cpu", batch_size=25)
print("[INFO] Running on device:", device)

gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'student_combined_epoch_148.pkl',
    arch='ResNet50',
    device=device
)
cudnn.enabled = True

def gaze_worker():
    global display_frame, video_writer
    while not rospy.is_shutdown():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            if frame is None or frame.size == 0:
                continue

            start_time = time.time()
            results = gaze_pipeline.step(frame)
            output = render(frame, results)
            fps = 1.0 / (time.time() - start_time)

            cv2.putText(output, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if video_writer is None:
                h, w = output.shape[:2]
                video_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(video_output_path, fourcc, 5, video_size)

            video_writer.write(output)

            with lock:
                display_frame = output

        except Exception as e:
            rospy.logerr(f"[Worker Error] {e}")

def callback(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if frame is not None and frame.size > 0 and not frame_queue.full():
            frame_queue.put_nowait(frame)
    except Exception as e:
        rospy.logerr(f"[Callback Error] {e}")

if __name__ == '__main__':
    rospy.init_node("gaze_estimation_node", anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, callback)
    rospy.loginfo("Running L2CS Gaze Estimation on /camera/color/image_raw")

    threading.Thread(target=gaze_worker, daemon=True).start()

    # === Matplotlib Setup ===
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))

    try:
        while not rospy.is_shutdown():
            with lock:
                if display_frame is not None:
                    ax.clear()
                    ax.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    ax.set_title("Gaze Estimation")
                    ax.axis("off")
                    plt.pause(0.001)
            time.sleep(0.03)

    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
    finally:
        plt.ioff()
        plt.close()

        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Video saved to {video_output_path}")

