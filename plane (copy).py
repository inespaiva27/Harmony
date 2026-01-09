# Requires: OpenCV, OpenPose Python API
# If OpenPose not installed to /usr/local/python, adjust sys.path below.

import os
import sys
import cv2
import math
from sys import platform
import numpy as np
import pathlib

# -------------------
# Tunables
# -------------------
SHOULDER_WIDTH_CM = 36.0   # avg biacromial breadth (for px/cm scaling)
PLANE_OFFSET_CM   = 30.0   # "plane" distance (cm) from elbow
PLANE_HYST_CM     = 2.0    # hysteresis (cm) to reduce flicker
CONF_THR          = 0.70   # keypoint confidence threshold
CAMERA_DEVICE     = "/dev/video6"  # change if needed
MODEL_FOLDER      = "/home/ines/Desktop/Harmony/openpose/models"
OPENPOSE_PYTHON   = "/home/ines/Desktop/Harmony/openpose/build/python"  # adjust if needed

# -------------------
# OpenPose import
# -------------------
try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if platform == "win32":
        sys.path.append(dir_path + '/../../python/openpose/Release')
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Add your built OpenPose python path
        sys.path.append(OPENPOSE_PYTHON)
        from openpose import pyopenpose as op
except ImportError as e:
    print("Error: OpenPose library could not be found. "
          "Check OPENPOSE_PYTHON path and that you enabled BUILD_PYTHON in CMake.")
    raise e

# -------------------
# Helper functions
# -------------------
def keypoint_ok(kp, conf=CONF_THR):
    return (kp[0] > 0) and (kp[1] > 0) and (kp[2] >= conf)

def get_shoulder_width_px(person, frame_width):
    """
    COCO-style indices used here:
      Right shoulder = 2, Left shoulder = 5
    Keypoints are normalized [0..1], convert to pixels using frame width.
    """
    rs = person[2]
    ls = person[5]
    if not (keypoint_ok(rs) and keypoint_ok(ls)):
        return None
    return abs((rs[0] - ls[0]) * frame_width)

def cm_to_px(cm, person, frame_width, shoulder_width_cm=SHOULDER_WIDTH_CM):
    sw_px = get_shoulder_width_px(person, frame_width)
    if sw_px is None or shoulder_width_cm <= 0:
        return None
    px_per_cm = sw_px / shoulder_width_cm
    return cm * px_per_cm

def elbows_and_wrists_px(person, frame_width, frame_height):
    """
    Return elbow and wrist pixel coords for both arms:
      {
        "Left":  ((ex, ey), (wx, wy)) or None,
        "Right": ((ex, ey), (wx, wy)) or None
      }
    COCO-style:
      Right elbow = 3, Right wrist = 4
      Left elbow  = 6, Left wrist  = 7
    """
    results = {}
    # Left arm
    l_el = person[6]; l_wr = person[7]
    if keypoint_ok(l_el) and keypoint_ok(l_wr):
        ex = l_el[0] * frame_width;  ey = l_el[1] * frame_height
        wx = l_wr[0] * frame_width;  wy = l_wr[1] * frame_height
        results["Left"] = ((ex, ey), (wx, wy))
    else:
        results["Left"] = None
    # Right arm
    r_el = person[3]; r_wr = person[4]
    if keypoint_ok(r_el) and keypoint_ok(r_wr):
        ex = r_el[0] * frame_width;  ey = r_el[1] * frame_height
        wx = r_wr[0] * frame_width;  wy = r_wr[1] * frame_height
        results["Right"] = ((ex, ey), (wx, wy))
    else:
        results["Right"] = None
    return results

def wrists_vs_plane_elbow(person, frame_width, frame_height,
                          plane_offset_cm=PLANE_OFFSET_CM,
                          shoulder_width_cm=SHOULDER_WIDTH_CM):
    """
    Elbow-anchored plane rule:
    delta_px = (elbow->wrist distance in px) - (plane_offset_cm in px)
      > 0  => wrist is beyond the elbow plane (extended)
      <= 0 => wrist is behind the elbow plane (retracted)
    Returns: {"Left": delta_px or None, "Right": delta_px or None}
    """
    plane_px = cm_to_px(plane_offset_cm, person, frame_width, shoulder_width_cm)
    if plane_px is None:
        return {"Left": None, "Right": None}

    results = {}
    arms = elbows_and_wrists_px(person, frame_width, frame_height)
    for side in ("Left", "Right"):
        if arms[side] is None:
            results[side] = None
            continue
        (ex, ey), (wx, wy) = arms[side]
        d_px = math.hypot(wx - ex, wy - ey)
        results[side] = d_px - plane_px
    return results

def defineIdPos(datums, image_width):
    """
    Return indices (left_id, right_id) based on average shoulder x-position.
    Special cases:
      - If exactly 1 person: (0, -1)
      - If >2 people: (0, -2)  (skip frame to avoid ambiguity)
      - If exactly 2: order by avg shoulder x (smaller x => 'Left')
    """
    datum = datums[0]
    if datum.poseKeypoints is None:
        return

    num_people = len(datum.poseKeypoints)
    if num_people == 1:
        return (0, -1)
    if num_people > 2:
        return (0, -2)

    RIGHT_SHOULDER = 2
    LEFT_SHOULDER  = 5

    def get_avg_shoulder_x(person):
        rs = person[RIGHT_SHOULDER]
        ls = person[LEFT_SHOULDER]
        rs_valid = rs[2] > CONF_THR
        ls_valid = ls[2] > CONF_THR
        coords = []
        if rs_valid: coords.append(rs[0] * image_width)
        if ls_valid: coords.append(ls[0] * image_width)
        if coords:
            return sum(coords) / len(coords)
        else:
            return None

    # Your OpenPose often gives person 0/1; compute avg shoulder x for both
    p0 = datum.poseKeypoints[1]
    p1 = datum.poseKeypoints[0]
    p0X = get_avg_shoulder_x(p0)
    p1X = get_avg_shoulder_x(p1)

    if p0X is None or p1X is None:
        return

    if p0X < p1X:
        return (0, 1)
    else:
        return (1, 0)

def draw_plane_and_vectors_elbow(img, person, frame_width, frame_height, who_label=None):
    """
    Visual overlay (elbow-anchored):
      - Plane: circle of radius plane_px around each elbow (≈30 cm in px)
      - Hysteresis band: plane ± hyst (thin circles)
      - Wrist→elbow line, distance, and EXT/RET label per arm
      - Optional 'who_label' ('Left person'/'Right person') near the torso
    """
    plane_px = cm_to_px(PLANE_OFFSET_CM, person, frame_width)
    if plane_px is None:
        return
    hyst_px  = cm_to_px(PLANE_HYST_CM, person, frame_width) or 0.0

    # If a label is requested, try to place near shoulder midpoint
    if who_label and keypoint_ok(person[2]) and keypoint_ok(person[5]):
        mx = int(((person[2][0] + person[5][0]) * 0.5) * frame_width)
        my = int(((person[2][1] + person[5][1]) * 0.5) * frame_height)
        cv2.putText(img, who_label, (mx + 8, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    # COCO indices: Right elbow=3, Right wrist=4 ; Left elbow=6, Left wrist=7
    for side, (e_idx, w_idx) in {"Left": (6, 7), "Right": (3, 4)}.items():
        e = person[e_idx]; w = person[w_idx]
        if not keypoint_ok(e):
            continue

        ex = int(e[0] * frame_width);  ey = int(e[1] * frame_height)

        # Plane circle at elbow
        cv2.circle(img, (ex, ey), int(plane_px), (0, 255, 255), 2)
        # Hysteresis band
        if hyst_px > 0:
            cv2.circle(img, (ex, ey), int(plane_px + hyst_px), (0, 200, 200), 1)
            inner = max(1, int(plane_px - hyst_px))
            cv2.circle(img, (ex, ey), inner, (0, 200, 200), 1)
        # Elbow crosshair
        cv2.drawMarker(img, (ex, ey), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)

        if keypoint_ok(w):
            wx = int(w[0] * frame_width);  wy = int(w[1] * frame_height)
            d_px = math.hypot(wx - ex, wy - ey)
            state = "EXT" if d_px > plane_px else "RET"
            color = (0, 255, 0) if state == "EXT" else (0, 0, 255)

            # Vector and wrist marker
            cv2.line(img, (ex, ey), (wx, wy), color, 2)
            cv2.circle(img, (wx, wy), 4, color, -1)

            txt = f"{side}: d={d_px:.0f}px thr={plane_px:.0f}px {state}"
            ty = ey - 10 if side == "Left" else ey + 20
            cv2.putText(img, txt, (ex + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# -------------------
# OpenPose setup
# -------------------
params = dict()
params["model_folder"]   = MODEL_FOLDER
params["net_resolution"] = "192x144"
params["keypoint_scale"] = "3"   # scaled to [0..1]
params["camera"]         = -1

print("Starting OpenPose...")
opWrapper = op.WrapperPython(op.ThreadManagerMode.Asynchronous)
opWrapper.configure(params)
opWrapper.start()

# -------------------
# Video capture
# -------------------
cap = cv2.VideoCapture(CAMERA_DEVICE)
if not cap.isOpened():
    print(f"Failed to open camera {CAMERA_DEVICE}")
    sys.exit(1)
else:
    print("Camera opened.")

# -------------------
# Main loop
# -------------------
# Separate state machines for Left person and Right person
checkStatus_left  = 0  # 0 -> waiting for EXTENDED; 1 -> waiting for RETRACTED
checkStatus_right = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame_height, frame_width = frame.shape[:2]

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Visuals (OpenPose rendered or raw)
    vis = datum.cvOutputData.copy() if datum.cvOutputData is not None else frame.copy()

    # Determine left/right participant indices
    idPos = defineIdPos([datum], frame_width)

    if idPos is None:
        # No people found
        cv2.imshow("OpenPose + Elbow Plane (2-person)", vis)
        if cv2.waitKey(1) == 27:
            break
        continue

    left_id, right_id = idPos

    # Handle special cases
    if right_id == -2:
        # >2 people — ambiguous; skip this frame (or implement your own selection)
        cv2.putText(vis, "More than 2 people detected — skipping",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2, cv2.LINE_AA)
    elif right_id == -1:
        # Exactly 1 person — treat them as 'Left person' and run single-person logic
        person = datum.poseKeypoints[left_id]
        draw_plane_and_vectors_elbow(vis, person, frame_width, frame_height, who_label="Left person")

        deltas = wrists_vs_plane_elbow(person, frame_width, frame_height)
        hyst_px = cm_to_px(PLANE_HYST_CM, person, frame_width) or 0.0

        if checkStatus_left == 0:
            for arm_side in ("Left", "Right"):
                d = deltas.get(arm_side)
                if d is not None and d > +hyst_px:
                    checkStatus_left = 1
                    print(f"Left person - {arm_side} Arm Extended (elbow plane)")
                    break
        else:
            for arm_side in ("Left", "Right"):
                d = deltas.get(arm_side)
                if d is not None and d < -hyst_px:
                    checkStatus_left = 0
                    print(f"Left person - {arm_side} Arm Retracted (elbow plane)")
                    break
    else:
        # Exactly 2 people — process both sides
        # LEFT PERSON
        personL = datum.poseKeypoints[left_id]
        draw_plane_and_vectors_elbow(vis, personL, frame_width, frame_height, who_label="Left person")
        deltasL = wrists_vs_plane_elbow(personL, frame_width, frame_height)
        hystL = cm_to_px(PLANE_HYST_CM, personL, frame_width) or 0.0

        if checkStatus_left == 0:
            for arm_side in ("Left", "Right"):
                d = deltasL.get(arm_side)
                if d is not None and d > +hystL:
                    checkStatus_left = 1
                    print(f"Left person - {arm_side} Arm Extended (elbow plane)")
                    break
        else:
            for arm_side in ("Left", "Right"):
                d = deltasL.get(arm_side)
                if d is not None and d < -hystL:
                    checkStatus_left = 0
                    print(f"Left person - {arm_side} Arm Retracted (elbow plane)")
                    break

        # RIGHT PERSON
        personR = datum.poseKeypoints[right_id]
        draw_plane_and_vectors_elbow(vis, personR, frame_width, frame_height, who_label="Right person")
        deltasR = wrists_vs_plane_elbow(personR, frame_width, frame_height)
        hystR = cm_to_px(PLANE_HYST_CM, personR, frame_width) or 0.0

        if checkStatus_right == 0:
            for arm_side in ("Left", "Right"):
                d = deltasR.get(arm_side)
                if d is not None and d > +hystR:
                    checkStatus_right = 1
                    print(f"Right person - {arm_side} Arm Extended (elbow plane)")
                    break
        else:
            for arm_side in ("Left", "Right"):
                d = deltasR.get(arm_side)
                if d is not None and d < -hystR:
                    checkStatus_right = 0
                    print(f"Right person - {arm_side} Arm Retracted (elbow plane)")
                    break

    # Show window
    cv2.imshow("OpenPose + Elbow Plane (2-person)", vis)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
