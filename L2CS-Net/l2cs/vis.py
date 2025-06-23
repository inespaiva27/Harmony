import cv2
import numpy as np
from .results import GazeResultContainer
from collections import deque

#class GazeHistory:
#    def __init__(self, maxlen=3):
#        self.pitch_buffer = deque(maxlen=maxlen)
#        self.yaw_buffer = deque(maxlen=maxlen)
#
#    def update(self, pitch, yaw):
#        self.pitch_buffer.append(pitch)
#        self.yaw_buffer.append(yaw)
#
#    def get_smoothed(self):
#        if len(self.pitch_buffer) == 0:
#            return None, None
#        smoothed_pitch = np.mean(self.pitch_buffer)
#        smoothed_yaw = np.mean(self.yaw_buffer)
#        return smoothed_pitch, smoothed_yaw
#    
#gaze_histories = {}


def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = c
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out

def draw_bbox(frame: np.ndarray, bbox: np.ndarray):
    
    x_min=int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min=int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max=int(bbox[2])
    y_max=int(bbox[3])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

    return frame

def is_mutual_gaze(pitch, yaw, angle_threshold):
    gaze = np.array([
        -np.cos(pitch) * np.sin(yaw),
        -0.5 * np.sin(pitch),  # weight pitch less
         np.cos(pitch) * np.cos(yaw)
    ])
    forward = np.array([0, 0, 1])
    cos_angle = np.dot(gaze, forward)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle_rad < angle_threshold



def render(frame: np.ndarray, results: GazeResultContainer):
    #global gaze_histories
    ANGLE_THRESHOLD = 0.2

    for i in range(results.pitch.shape[0]):
        bbox = results.bboxes[i]
        pitch = results.pitch[i]
        yaw = results.yaw[i]

        x_min = max(int(bbox[0]), 0)
        y_min = max(int(bbox[1]), 0)
        x_max = int(bbox[2])
        y_max = int(bbox[3])
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        draw_bbox(frame, bbox)

        # Draw mutual gaze detection region as a circle
        center_x = int(x_min + bbox_width / 2)
        center_y = int(y_min + bbox_height / 2)

        # Radius based on ANGLE_THRESHOLD and bbox size (tweak scale as needed)
        # Empirically scale to image: more threshold → bigger radius
        radius = int((bbox_width / 2) * np.tan(ANGLE_THRESHOLD))
 

        cv2.circle(frame, (center_x, center_y), radius, (100, 255, 100), 1, cv2.LINE_AA)


        # Get or create history buffer
        #if i not in gaze_histories:
        #    gaze_histories[i] = GazeHistory(maxlen=5)
        #gaze_histories[i].update(pitch, yaw)
        #smooth_pitch, smooth_yaw = gaze_histories[i].get_smoothed()

        # Compute mutual gaze using smoothed values
        if is_mutual_gaze(pitch, yaw, angle_threshold=ANGLE_THRESHOLD):
            arrow_color = (0, 255, 0)
        else:
            arrow_color = (0, 0, 255)

        draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (pitch, yaw), color=arrow_color)

    return frame

