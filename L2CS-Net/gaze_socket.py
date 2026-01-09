import zmq
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
frame_queue = queue.Queue(maxsize=1)
display_frame = None
lock = threading.Lock()

video_writer = None
video_output_path = str(CWD / 'output_gaze_estimation.avi')
video_size = (640, 480)

# === Load Gaze Pipeline ===
device = select_device("gpu:0" if torch.cuda.is_available() else "cpu", batch_size=25)
print("[INFO] Running on device:", device)

gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'student_combined_epoch_148.pkl',
    arch='ResNet50',
    device=device
)
cudnn.enabled = True

# === ZMQ Setup ===
context = zmq.Context()
frame_socket = context.socket(zmq.PULL)
frame_socket.connect("tcp://localhost:5556")  # match your sender

def gaze_worker():
    global display_frame, video_writer
    while True:
        try:
            # Receive JPEG-encoded frame
            msg = frame_socket.recv()
            np_arr = np.frombuffer(msg, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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
            print(f"[Worker Error] {e}")
            time.sleep(0.1)

if __name__ == '__main__':
    print("Running L2CS Gaze Estimation via Socket Stream")

    threading.Thread(target=gaze_worker, daemon=True).start()

    # === Matplotlib Setup ===
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))

    try:
        while True:
            with lock:
                if display_frame is not None:
                    ax.clear()
                    ax.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    ax.set_title("Gaze Estimation")
                    ax.axis("off")
                    plt.pause(0.001)
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("Shutting down viewer.")
    finally:
        plt.ioff()
        plt.close()

        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Video saved to {video_output_path}")
