# -*- coding: utf-8 -*-
# OpenPose + RealSense depth + L2CS Gaze
# Shows two separate windows:
#   - "OpenPose + Depth Plane": skeleton + depth-plane arm states
#   - "L2CS Gaze": gaze overlay only
#
# States: EXTENDED (green) / RETRACTED (red)

import os
import sys
import math
import csv
from datetime import datetime
import numpy as np
import cv2
from sys import platform
import argparse
import pathlib

# ---- Gaze (L2CS) ----
import torch
import torch.backends.cudnn as cudnn
from L2CS_Net.l2cs import select_device, Pipeline, render
from L2CS_Net.l2cs.results import GazeResultContainer

# ---- RealSense ----
try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 not found. Install Intel RealSense SDK (librealsense).")
    raise

# -------------------
# Tunables
# -------------------
PLANE_OFFSET_M   = 0.30   # plane 30 cm in front (toward camera => smaller Z)
PLANE_HYST_M     = 0.02   # 2 cm hysteresis
CONF_THR         = 0.70   # OpenPose keypoint confidence
MODEL_FOLDER     = "/home/ines/Desktop/Harmony/openpose/models"
OPENPOSE_PYTHON  = "/home/ines/Desktop/Harmony/openpose/build/python"
PLANE_REF_MODE   = "torso"    # 'torso' | 'shoulder' | 'elbow'
GAZE_EVERY       = 4          # run gaze every N frames

# -------------------
# Imports (OpenPose)
# -------------------
try:
    if platform == "win32":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(dir_path + '/../../python/openpose/Release')
        os.environ['PATH'] += ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        sys.path.append(OPENPOSE_PYTHON)
        from openpose import pyopenpose as op
except ImportError as e:
    print("OpenPose Python API not found. Check OPENPOSE_PYTHON and BUILD_PYTHON.")
    raise e

# -------------------
# Helpers (OpenPose + depth)
# -------------------
def keypoint_ok(kp, conf=CONF_THR):
    return (kp[0] > 0) and (kp[1] > 0) and (kp[2] >= conf)

def median_depth_m(depth_frame, u, v, win=2):
    w = depth_frame.get_width()
    h = depth_frame.get_height()
    vals = []
    for dy in range(-win, win+1):
        yy = min(max(v+dy, 0), h-1)
        for dx in range(-win, win+1):
            xx = min(max(u+dx, 0), w-1)
            d = depth_frame.get_distance(xx, yy)
            if d > 0:
                vals.append(d)
    return float(np.median(vals)) if vals else 0.0

def elbows_shoulders_wrists_px(person, W, H):
    out = {"Left": {"S": None, "E": None, "W": None},
           "Right": {"S": None, "E": None, "W": None}}
    Ls, Le, Lw = person[5], person[6], person[7]
    Rs, Re, Rw = person[2], person[3], person[4]
    if keypoint_ok(Ls): out["Left"]["S"] = (int(Ls[0]*W), int(Ls[1]*H))
    if keypoint_ok(Le): out["Left"]["E"] = (int(Le[0]*W), int(Le[1]*H))
    if keypoint_ok(Lw): out["Left"]["W"] = (int(Lw[0]*W), int(Lw[1]*H))
    if keypoint_ok(Rs): out["Right"]["S"] = (int(Rs[0]*W), int(Rs[1]*H))
    if keypoint_ok(Re): out["Right"]["E"] = (int(Re[0]*W), int(Re[1]*H))
    if keypoint_ok(Rw): out["Right"]["W"] = (int(Rw[0]*W), int(Rw[1]*H))
    return out

def torso_center_px(person, W, H):
    Rs, Ls = person[2], person[5]
    if keypoint_ok(Rs) and keypoint_ok(Ls):
        x = int(((Rs[0] + Ls[0]) * 0.5) * W)
        y = int(((Rs[1] + Ls[1]) * 0.5) * H)
        return (x, y)
    return None

def defineIdPos(datum, image_width):
    if datum.poseKeypoints is None:
        return None
    n = len(datum.poseKeypoints)
    if n == 0: return None
    if n == 1: return (0, -1)
    if n > 2:  return (0, -2)

    RIGHT_SHOULDER, LEFT_SHOULDER = 2, 5
    def avg_sh_x(p):
        rs, ls = p[RIGHT_SHOULDER], p[LEFT_SHOULDER]
        xs = []
        if rs[2] >= CONF_THR: xs.append(rs[0]*image_width)
        if ls[2] >= CONF_THR: xs.append(ls[0]*image_width)
        return sum(xs)/len(xs) if xs else None

    p0, p1 = datum.poseKeypoints[0], datum.poseKeypoints[1]
    x0, x1 = avg_sh_x(p0), avg_sh_x(p1)
    if x0 is None or x1 is None:
        return None
    return (0, 1) if x0 < x1 else (1, 0)

def arm_status_color(status):
    return (0, 255, 0) if status == 'EXT' else (0, 0, 255)

def draw_status_overlay(img, ex, ey, wx, wy, status, plane_z, wrist_z):
    color = arm_status_color(status)
    cv2.line(img, (ex, ey), (wx, wy), color, 3)
    cv2.circle(img, (wx, wy), 5, color, -1)
    cv2.putText(img, status, (wx+6, wy-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"w:{wrist_z:.2f} p:{plane_z:.2f}", (wx+6, wy+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,230,230), 1, cv2.LINE_AA)

def plane_reference_z(depth, person, parts, side, W, H):
    if PLANE_REF_MODE == "torso":
        tc = torso_center_px(person, W, H)
        if tc is None: return None
        tz = median_depth_m(depth, tc[0], tc[1])
        return tz - PLANE_OFFSET_M if tz > 0 else None
    elif PLANE_REF_MODE == "shoulder":
        S = parts[side]["S"]
        if S is None: return None
        sz = median_depth_m(depth, S[0], S[1])
        return sz - PLANE_OFFSET_M if sz > 0 else None
    else:
        E = parts[side]["E"]
        if E is None: return None
        ez = median_depth_m(depth, E[0], E[1])
        return ez - PLANE_OFFSET_M if ez > 0 else None

# -------------------
# Helpers for gaze association (same logic as server)
# -------------------
def head_anchor_px(person, W, H, conf_thr=CONF_THR):
    nose = person[0] if len(person) > 0 else None
    neck = person[1] if len(person) > 1 else None
    rs   = person[2] if len(person) > 2 else None
    ls   = person[5] if len(person) > 5 else None

    def ok(kp): return (kp is not None) and (kp[0] > 0) and (kp[1] > 0) and (kp[2] >= conf_thr)
    if ok(nose):  return (int(nose[0]*W), int(nose[1]*H))
    if ok(neck):  return (int(neck[0]*W), int(neck[1]*H))
    if ok(rs) and ok(ls):
        x = int(((rs[0]+ls[0])*0.5)*W)
        y = int(((rs[1]+ls[1])*0.5)*H)
        return (x, y)
    return None

def bbox_center_x(b):
    if b is None:
        return None
    if isinstance(b, dict):
        return 0.5 * (b['x1'] + b['x2'])
    x1, y1, x2, y2 = b
    return 0.5 * (x1 + x2)

def normalize_gaze_results(gr):
    """
    Convert your GazeResultContainer (Pipeline.step output)
    into a list of detection dicts with yaw/pitch in *degrees*
    plus bbox and score.

    Each output element looks like:
      {
        "yaw_deg":   float,
        "pitch_deg": float,
        "bbox":      [x1, y1, x2, y2] or None,
        "score":     float or None
      }
    """
    if gr is None:
        return []

    # --- Case 1: your Pipeline's result container ---
    # Either check exact type...
    if isinstance(gr, GazeResultContainer) or (
        hasattr(gr, "pitch") and hasattr(gr, "yaw") and hasattr(gr, "bboxes")
    ):
        # Ensure 1D arrays
        pitch_arr = np.array(gr.pitch).reshape(-1)   # radians
        yaw_arr   = np.array(gr.yaw).reshape(-1)     # radians

        # These might be empty if no faces
        try:
            bboxes = np.array(gr.bboxes)
        except Exception:
            bboxes = np.zeros((0, 4), dtype=float)

        try:
            scores = np.array(gr.scores)
        except Exception:
            scores = None

        N = pitch_arr.shape[0]
        out = []
        for i in range(N):
            yaw_rad   = float(yaw_arr[i])
            pitch_rad = float(pitch_arr[i])

            # radians → degrees
            yaw_deg   = math.degrees(yaw_rad)
            pitch_deg = math.degrees(pitch_rad)

            box = None
            if bboxes.ndim == 2 and bboxes.shape[0] > i:
                box = bboxes[i]
                box = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]

            score = None
            if scores is not None and scores.ndim >= 1 and scores.shape[0] > i:
                score = float(scores[i])

            out.append({
                "yaw_deg":   yaw_deg,
                "pitch_deg": pitch_deg,
                "bbox":      box,
                "score":     score,
            })
        return out

    # --- Case 2: already a list/tuple of dicts/objects (fallback) ---
    if isinstance(gr, (list, tuple)):
        return list(gr)

    # --- Case 3: dict wrapper (faces/detections etc.) ---
    if isinstance(gr, dict):
        for key in ("detections", "faces"):
            if key in gr and isinstance(gr[key], (list, tuple)):
                return list(gr[key])
        return []

    # --- Other unexpected types → empty ---
    return []


def get_det_value(det, keys, default=None):
    """
    Reads a field from either a dict or object by trying several key/attr names.
    """
    if isinstance(det, dict):
        for k in keys:
            if k in det:
                return det[k]
        return default
    for k in keys:
        if hasattr(det, k):
            return getattr(det, k)
    return default

def yawpitch_to_vec(yaw_deg, pitch_deg):
    """
    Convert yaw/pitch (deg) to a unit vector in the camera frame.
    Same convention as in your server code.
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    vx =  math.cos(p) * math.sin(y)
    vy = -math.sin(p)
    vz =  math.cos(p) * math.cos(y)
    n = (vx*vx + vy*vy + vz*vz) ** 0.5 or 1.0
    return (vx/n, vy/n, vz/n)

# -------------------
# Args + setups
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", type=str,
                    help="Device to run model: cpu or gpu:0")
parser.add_argument("--snapshot",
    default="/home/ines/Desktop/Harmony/L2CS_Net/models/student_combined_epoch_148.pkl",
    type=str, help="Path to L2CS weights")
parser.add_argument("--arch", default="ResNet50", type=str,
                    help="Network architecture (ResNet18/34/50...)")
args, unknown = parser.parse_known_args()

# OpenPose
params = {"model_folder": MODEL_FOLDER, "net_resolution": "192x144", "keypoint_scale": "3"}
print("Starting OpenPose…")
opWrapper = op.WrapperPython(op.ThreadManagerMode.Asynchronous)
opWrapper.configure(params)
opWrapper.start()

# L2CS
print("Loading L2CS…")
CWD = pathlib.Path.cwd()
cudnn.enabled = True

weights_path = pathlib.Path(args.snapshot)
if not weights_path.exists():
    # fallback if relative path
    weights_path = CWD / 'L2CS_Net' / 'models' / 'student_combined_epoch_148.pkl'

gaze_pipeline = Pipeline(
    weights=weights_path,
    arch=args.arch,
    device=select_device(args.device, batch_size=1)
)

# RealSense
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)
align_to_color = rs.align(rs.stream.color)

state = {"left": 0, "right": 0}

# CSV logging setup
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)
run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(logs_dir, f"gaze_log_{run_ts}.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp", "frame_idx", "person_id", "label",
    "yaw_deg", "pitch_deg",
    "vx", "vy", "vz",
    "cx"
])

print(f"Gaze CSV logging to: {csv_path}")
print("Running (ESC to quit)…")

frame_idx = 0
gaze_frame_cached = None

try:
    while True:
        frames = pipe.wait_for_frames()
        frames = align_to_color.process(frames)
        depth  = frames.get_depth_frame()
        color  = frames.get_color_frame()
        if not depth or not color:
            continue
        color_img = np.asanyarray(color.get_data())
        H, W = color_img.shape[:2]

        # --- OpenPose every frame
        datum = op.Datum()
        datum.cvInputData = color_img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        vis = datum.cvOutputData.copy() if datum.cvOutputData is not None else color_img.copy()

        # --- identify left/right persons (for both arm and gaze)
        idpos = defineIdPos(datum, W)
        persons = []
        if idpos is not None and datum.poseKeypoints is not None:
            left_id, right_id = idpos
            if right_id == -2:
                cv2.putText(vis, "More than 2 people — skipping", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 255), 2, cv2.LINE_AA)
            elif right_id == -1:
                person = datum.poseKeypoints[left_id]
                persons = [("left", "Left person", person)]
            else:
                persons = [
                    ("left", "Left person", datum.poseKeypoints[left_id]),
                    ("right", "Right person", datum.poseKeypoints[right_id])
                ]

        # --- L2CS every N frames (same logic as demo.py)
        if (frame_idx % GAZE_EVERY) == 0 or gaze_frame_cached is None:
            with torch.no_grad():
                gaze_results = gaze_pipeline.step(color_img)    # <- SAME as demo.py
            gaze_frame_cached = render(color_img.copy(), gaze_results)

            # --- Build gaze logs for this frame
            detections = normalize_gaze_results(gaze_results)

            # Build head anchors (for assignment)
            anchors = []
            for who_key, who_label, person in persons:
                ha = head_anchor_px(person, W, H)
                if ha is not None:
                    anchors.append((who_key, who_label, ha[0]))

            # Associate detections to nearest head in X
            for det in detections:
                yaw   = get_det_value(det, ["yaw", "yaw_deg"], 0.0)
                pitch = get_det_value(det, ["pitch", "pitch_deg"], 0.0)
                bbox  = get_det_value(det, ["bbox", "box", "rect"], None)
                cx    = bbox_center_x(bbox)

                try:
                    yaw = float(yaw)
                except Exception:
                    yaw = 0.0
                try:
                    pitch = float(pitch)
                except Exception:
                    pitch = 0.0

                vx, vy, vz = yawpitch_to_vec(yaw, pitch)

                person_id = "none"
                label = "Unassigned"
                if anchors and (cx is not None):
                    who_key, who_label, _ = min(anchors, key=lambda a: abs(cx - a[2]))
                    person_id = who_key
                    label = who_label

                ts = datetime.utcnow().isoformat() + "Z"
                csv_writer.writerow([
                    ts, frame_idx, person_id, label,
                    yaw, pitch,
                    vx, vy, vz,
                    cx if cx is not None else ""
                ])

        # ---- Depth-plane logic (arm extended/retracted) on vis
        if idpos is not None and datum.poseKeypoints is not None:
            left_id, right_id = idpos
            if right_id != -2:  # only if <=2 people
                for who_key, who_label, person in persons:
                    parts = elbows_shoulders_wrists_px(person, W, H)
                    current_status = 'EXT' if state[who_key] == 1 else 'RET'
                    for side in ("Left", "Right"):
                        E = parts[side]["E"]; Wp = parts[side]["W"]
                        if Wp is None or E is None:
                            continue
                        wx, wy = Wp; ex, ey = E
                        wrist_z = median_depth_m(depth, wx, wy)
                        if wrist_z <= 0:
                            draw_status_overlay(vis, ex, ey, wx, wy, current_status, 0.0, 0.0)
                            continue
                        plane_z = plane_reference_z(depth, person, parts, side, W, H)
                        if plane_z is None:
                            draw_status_overlay(vis, ex, ey, wx, wy, current_status, 0.0, wrist_z)
                            continue
                        if state[who_key] == 0:
                            if wrist_z < (plane_z - PLANE_HYST_M):
                                state[who_key] = 1
                                current_status = 'EXT'
                                print(f"{who_label} - {side} arm EXTENDED (w={wrist_z:.2f}, p={plane_z:.2f})")
                        else:
                            if wrist_z > (plane_z + PLANE_HYST_M):
                                state[who_key] = 0
                                current_status = 'RET'
                                print(f"{who_label} - {side} arm RETRACTED (w={wrist_z:.2f}, p={plane_z:.2f})")
                        draw_status_overlay(vis, ex, ey, wx, wy, current_status, plane_z, wrist_z)

        # --- Display: separate windows
        cv2.imshow("OpenPose + Depth Plane", vis)
        cv2.imshow("L2CS Gaze", gaze_frame_cached)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        frame_idx += 1

finally:
    pipe.stop()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Gaze CSV saved at: {csv_path}")

