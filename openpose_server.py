# From Python
# It requires OpenCV installed for Python

import os
import sys
import cv2
import zmq
import av
import math
import numpy as np
import argparse
import websocket
import pathlib
import logging
from datetime import datetime
from sys import platform
import json



# ---- RealSense (depth) ----
try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 not found. Install Intel RealSense SDK (librealsense).")
    raise

# ---- Gaze (L2CS) ----
import torch
import torch.backends.cudnn as cudnn
from L2CS_Net.l2cs import select_device, Pipeline, render
from L2CS_Net.l2cs.results import GazeResultContainer


# -------------------
# Tunables (depth-plane)
# -------------------
PLANE_OFFSET_M   = 0.30   # plane is 20 cm in front (toward camera => smaller Z)
PLANE_HYST_M     = 0.02   # 2 cm hysteresis
CONF_THR         = 0.70   # OpenPose keypoint confidence
MODEL_FOLDER     = "/home/ines/Desktop/Harmony/openpose/models"
OPENPOSE_PYTHON  = "/home/ines/Desktop/Harmony/openpose/build/python"  # adjust if needed

# Plane reference mode: 'torso' | 'shoulder' | 'elbow'
PLANE_REF_MODE   = "torso"

# -------------------
# Tunables (gaze)
# -------------------

GAZE_EVERY = 8

# Small debounce to avoid one-frame flips
STREAK_N = 2  # require 2 consecutive frames
streak = {
    "left_ext": 0, "left_ret": 0,
    "right_ext": 0, "right_ret": 0,
}


# --- Shared timestamped log directory ---
if "RUN_TIMESTAMP" not in os.environ:
    os.environ["RUN_TIMESTAMP"] = datetime.now().strftime("%Y%m%d_%H%M")

timestamp = os.environ["RUN_TIMESTAMP"]
log_dir = os.path.join(os.path.dirname(__file__), "logs", timestamp)
os.makedirs(log_dir, exist_ok=True)

# --- Logging setup ---
log_filename = os.path.join(log_dir, f"kinova_log_openpose_{timestamp}.log")
logger = logging.getLogger('openpose')
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# -------------------
# ZMQ server
# -------------------
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
logger.info("OpenPose server started and bound to tcp://*:5555")

pub = context.socket(zmq.PUB)          # ASYNC: gaze stream
pub.bind("tcp://*:5556")
logger.info("OpenPose server started and bound to tcp://*:5556")

# -------------------
# Helpers (keypoints + depth)
# -------------------
def keypoint_ok(kp, conf=CONF_THR):
    return (kp[0] > 0) and (kp[1] > 0) and (kp[2] >= conf)

def median_depth_m(depth_frame, u, v, win=2):
    """Median depth (m) in a (2*win+1)^2 window around (u,v). Ignores zeros."""
    w = depth_frame.get_width()
    h = depth_frame.get_height()
    vals = []
    for dy in range(-win, win+1):
        yy = min(max(v+dy, 0), h-1)
        for dx in range(-win, win+1):
            xx = min(max(u+dx, 0), w-1)
            d = depth_frame.get_distance(xx, yy)  # meters
            if d > 0:
                vals.append(d)
    if not vals:
        return 0.0
    return float(np.median(vals))

def elbows_shoulders_wrists_px(person, W, H):
    """
    Return pixel coords for shoulders, elbows, wrists:
      { "Left": {"S":(sx,sy)|None, "E":(ex,ey)|None, "W":(wx,wy)|None}, "Right": {...} }
    COCO: R-shoulder=2, R-elbow=3, R-wrist=4 ; L-shoulder=5, L-elbow=6, L-wrist=7
    """
    out = {"Left": {"S": None, "E": None, "W": None},
           "Right":{"S": None, "E": None, "W": None}}
    # Left
    Ls, Le, Lw = person[5], person[6], person[7]
    if keypoint_ok(Ls): out["Left"]["S"] = (int(Ls[0]*W), int(Ls[1]*H))
    if keypoint_ok(Le): out["Left"]["E"] = (int(Le[0]*W), int(Le[1]*H))
    if keypoint_ok(Lw): out["Left"]["W"] = (int(Lw[0]*W), int(Lw[1]*H))
    # Right
    Rs, Re, Rw = person[2], person[3], person[4]
    if keypoint_ok(Rs): out["Right"]["S"] = (int(Rs[0]*W), int(Rs[1]*H))
    if keypoint_ok(Re): out["Right"]["E"] = (int(Re[0]*W), int(Re[1]*H))
    if keypoint_ok(Rw): out["Right"]["W"] = (int(Rw[0]*W), int(Rw[1]*H))
    return out

def torso_center_px(person, W, H):
    """Midpoint between both shoulders; returns (x,y) or None."""
    Rs = person[2]; Ls = person[5]
    if keypoint_ok(Rs) and keypoint_ok(Ls):
        x = int(((Rs[0] + Ls[0]) * 0.5) * W)
        y = int(((Rs[1] + Ls[1]) * 0.5) * H)
        return (x, y)
    return None

def plane_reference_z(depth, person, parts, side, W, H):
    """
    plane_z for a wrist on 'side' based on PLANE_REF_MODE:
      'torso'    -> z(torso_center)   - 0.20
      'shoulder' -> z(shoulder side)  - 0.20
      'elbow'    -> z(elbow side)     - 0.20
    """
    if PLANE_REF_MODE == "torso":
        tc = torso_center_px(person, W, H)
        if tc is None: return None
        tz = median_depth_m(depth, tc[0], tc[1], win=2)
        if tz <= 0: return None
        return tz - PLANE_OFFSET_M
    elif PLANE_REF_MODE == "shoulder":
        S = parts[side]["S"]
        if S is None: return None
        sz = median_depth_m(depth, S[0], S[1], win=2)
        if sz <= 0: return None
        return sz - PLANE_OFFSET_M
    else:  # elbow
        E = parts[side]["E"]
        if E is None: return None
        ez = median_depth_m(depth, E[0], E[1], win=2)
        if ez <= 0: return None
        return ez - PLANE_OFFSET_M

def arm_status_color(status):
    # BGR: EXT -> green, RET -> red
    return (0, 255, 0) if status == 'EXT' else (0, 0, 255)

def draw_status_overlay(img, ex, ey, wx, wy, status, plane_z, wrist_z, label_at="wrist"):
    """Draw elbow crosshair, line to wrist colored by status, and small texts."""
    color = arm_status_color(status)
    cv2.drawMarker(img, (ex, ey), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
    cv2.line(img, (ex, ey), (wx, wy), color, 3)
    cv2.circle(img, (wx, wy), 5, color, -1)

    tx, ty = (wx + 6, wy - 6) if label_at == "wrist" else (ex + 6, ey - 6)
    bx, by = (wx + 6, wy + 16) if label_at == "wrist" else (ex + 6, ey + 16)

    cv2.putText(img, status, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"w:{wrist_z:.2f} p:{plane_z:.2f}", (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

def draw_person_label(img, person, W, H, text):
    if keypoint_ok(person[2]) and keypoint_ok(person[5]):
        mx = int(((person[2][0] + person[5][0]) * 0.5) * W)
        my = int(((person[2][1] + person[5][1]) * 0.5) * H)
        cv2.putText(img, text, (mx + 8, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
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
    if b is None: return None
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

    Falls back to list/dict handling if the type is not the container.
    """
    if gr is None:
        return []

    # --- Case 1: your Pipeline's result container ---
    if isinstance(gr, GazeResultContainer) or (
        hasattr(gr, "pitch") and hasattr(gr, "yaw") and hasattr(gr, "bboxes")
    ):
        # Ensure 1D arrays (radians)
        pitch_arr = np.array(gr.pitch).reshape(-1)
        yaw_arr   = np.array(gr.yaw).reshape(-1)

        # bboxes / scores may be empty if no faces
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
                b = bboxes[i]
                box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]

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

    # --- Case 2: already list/tuple of detections ---
    if isinstance(gr, (list, tuple)):
        return list(gr)

    # --- Case 3: dict wrapper (faces/detections) ---
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
    Convention (matches common L2CS usage + draw_gaze):
      - +yaw: subject looks to THEIR left → camera +X (image right is +X)
      - +pitch: subject looks DOWN → camera +Y down is positive, so we use -Y for 'up'
      - +Z points forward from the camera into the scene
    Adjust signs if your visualization shows flips.
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    vx =  math.cos(p) * math.sin(y)   # right (+X) when yaw left is positive
    vy = -math.sin(p)                 # up is -Y
    vz =  math.cos(p) * math.cos(y)   # forward (+Z)
    # Normalize (defensive)
    n = (vx*vx + vy*vy + vz*vz) ** 0.5 or 1.0
    return (vx/n, vy/n, vz/n)



# -------------------
# Person ordering helper (unchanged logic)
# -------------------


def defineIdPos(datums, image_width):
    datum = datums[0]
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

#def connect_kinova_camera(ws_url):
#    ws = websocket.create_connection(ws_url)
#    container = av.open(ws.makefile('rb'), format='h264')
#    return container

# -------------------
# OpenPose import
# -------------------

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('/home/ines/Desktop/Harmony/openpose/build/python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        logger.error(f'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder? {e}')
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

# -------------------
# Args L2CS-Net
# -------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", action="store_true", help="Disable display.")
    parser.add_argument('--device', help='Device to run model: cpu or gpu:0', default="cpu", type=str)
    parser.add_argument('--snapshot', help='Path of model snapshot.',
                        default='/home/ines/Desktop/Harmony/L2CS_Net/models/student_combined_epoch_148.pkl', type=str)
    parser.add_argument('--arch', help='Network architecture',
                        default='ResNet50', type=str)
    args, unknown = parser.parse_known_args()

# -------------------
# OpenPose params
# -------------------

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/home/ines/Desktop/Harmony/openpose/models"
    params["net_resolution"] = "192x144"
    params["keypoint_scale"] = "3"
    params["camera"] = -1

    

    for i in range(len(unknown)):
        curr_item = unknown[i]
        next_item = unknown[i + 1] if i + 1 < len(unknown) else "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = next_item

# -------------------
# Start OpenPose
# -------------------

    # Starting OpenPose
    logger.info("Starting OpenPose")
    print(" Starting OpenPose...")
    opWrapper = op.WrapperPython(op.ThreadManagerMode.Asynchronous)
    opWrapper.configure(params)
    opWrapper.start()

# -------------------
# Start Gaze (L2CS)
# ------------------- 

    CWD = pathlib.Path.cwd()
    cudnn.enabled = True

    weights_path = pathlib.Path(args.snapshot)
    if not weights_path.exists():
        weights_path = CWD / 'L2CS_Net' / 'models' / 'student_combined_epoch_148.pkl'
  
    gaze_pipeline = Pipeline(
        weights=weights_path,
        arch=args.arch,
        device=select_device(args.device, batch_size=1)
    )

  
# -------------------
# Start RealSense (aligned depth->color)
# -------------------
    print("Starting RealSense…")
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align_to_color = rs.align(rs.stream.color)

    # Hysteresis state per tracked person label (left/right in your semantics)
    # 0 => currently RET; 1 => currently EXT
    state = {"left": 0, "right": 0}

    while True:

        logger.info("OpenPose initialized - waiting for turn_order")
        print(" OpenPose initialized — waiting for turn_order...")
        turn_order_aux = socket.recv()
        logger.info(f"Message received: {turn_order_aux.decode()}") 
        print("Message received:", turn_order_aux)
        socket.send_string("Order Received")
        turn_order = turn_order_aux.decode()
        print(turn_order)
        turn_n = 0        

        # Main loop
        userWantsToExit = False
        checkStatus = 0
        buffer = []
        buffer_len = 0
        buffer_it = 0
        config_finished = False

        last_detected_count = -1

        frame_idx = 0
        gaze_frame_cached = None
        gaze_results_cached = None


        while not userWantsToExit:
            # --- Get synchronized color+depth
            frames = pipe.wait_for_frames()
            frames = align_to_color.process(frames)
            depth  = frames.get_depth_frame()
            color  = frames.get_color_frame()
            if not depth or not color:
                continue
            frame = np.asanyarray(color.get_data())
            H, W = frame.shape[:2]


            # --- OpenPose
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            datumProcessed = [datum]

            # --- Gaze (L2CS)
            if (frame_idx % GAZE_EVERY) == 0 or gaze_frame_cached is None:
                with torch.no_grad():
                    gaze_results = gaze_pipeline.step(frame)         # same as demo.py
                gaze_results_cached = gaze_results                   # store for ZMQ
                gaze_frame_cached = render(frame.copy(), gaze_results)   # same as demo.py




            # --- Person count debug
            current_count = len(datum.poseKeypoints) if datum.poseKeypoints is not None else 0
            if current_count != last_detected_count:
                last_detected_count = current_count
         
            # --- ID left/right by x (your helper)
            idPos = defineIdPos(datumProcessed, W)

            if not args.no_display:
                try:
                    frame_with_openpose = datum.cvOutputData if datum.cvOutputData is not None else frame
                    cv2.imshow("OpenPose + Depth Plane", frame_with_openpose)
                    cv2.imshow("L2CS Gaze", gaze_frame_cached)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        userWantsToExit = True
                except Exception as e:
                    logger.error(f"Display exception: {e}")
                    print(f"Display exception: {e}")

                    
            # --- Build ZMQ gaze payload on gaze frames
            try:
                if ((frame_idx % GAZE_EVERY) == 0) and (gaze_results_cached is not None):
                    gaze_payload = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "frame_idx": int(frame_idx),
                        "people": []
                    }
            
                    # Prepare person anchors if we have a definite left/right
                    persons = []
                    if (idPos is not None) and (idPos[1] not in (-1, -2)) and (datum.poseKeypoints is not None):
                        left_id, right_id = idPos
                        kp = datum.poseKeypoints
                        if 0 <= left_id < len(kp) and 0 <= right_id < len(kp):
                            persons = [
                                ("left",  "Left person",  kp[left_id]),
                                ("right", "Right person", kp[right_id]),
                            ]
            
                    # Anchor list: (who_key, who_label, head_center_x)
                    anchors = []
                    for who_key, who_label, person in persons:
                        ha = head_anchor_px(person, W, H)
                        if ha is not None:
                            anchors.append((who_key, who_label, ha[0]))
            
                    assigned = {}
            
                    # Normalize gaze results container → iterable of detections
                    detections = normalize_gaze_results(gaze_results_cached)
                    for det in detections:
                        yaw   = get_det_value(det, ["yaw", "yaw_deg"], 0.0)
                        pitch = get_det_value(det, ["pitch", "pitch_deg"], 0.0)
                        bbox  = get_det_value(det, ["bbox", "box", "rect"], None)
                        cx    = bbox_center_x(bbox)
            
                        # Defensive casts
                        try:    yaw = float(yaw)
                        except: yaw = 0.0
                        try:    pitch = float(pitch)
                        except: pitch = 0.0
            
                        # 3D direction in camera frame (unit vector)
                        vx, vy, vz = yawpitch_to_vec(yaw, pitch)
            
                        entry = {
                            "yaw_deg":   yaw,
                            "pitch_deg": pitch,
                            "bbox":      bbox,
                            "id":        None,
                            "label":     None,
                            "cx":        cx,
                            "dir_cam":   [vx, vy, vz]   # <-- include 3D vector
                        }
            
                        # Associate to nearest person (by X) if anchors exist
                        if anchors and (cx is not None):
                            who_key, who_label, _ = min(anchors, key=lambda a: abs(cx - a[2]))
                            entry["id"] = who_key
                            entry["label"] = who_label
                            assigned.setdefault(who_key, []).append(entry)
                        else:
                            gaze_payload["people"].append(entry)
            
                    # Emit in left/right order if we assigned
                    if assigned:
                        for who_key, _, _ in anchors:
                            for e in assigned.get(who_key, []):
                                gaze_payload["people"].append(e)
            
                    # Per-person 3D vector logs (one line per person)
                    for e in gaze_payload["people"]:
                        v = e.get("dir_cam") or [float('nan')]*3
                        logger.info(
                            "Gaze3D frame=%d id=%s label=%s v=[%.3f, %.3f, %.3f] yaw=%.1f pitch=%.1f cx=%s",
                            frame_idx, e.get("id"), e.get("label"),
                            v[0], v[1], v[2],
                            e.get("yaw_deg"), e.get("pitch_deg"),
                            str(e.get("cx"))
                        )
            
                    # Concise publish summary
                    logger.info("Gaze PUB: people=%d, frame_idx=%d",
                                len(gaze_payload["people"]), frame_idx)
            
                    # Publish once per gaze frame (non-blocking)
                    try:
                        pub.send_multipart(
                            [b"gaze", json.dumps(gaze_payload).encode("utf-8")],
                            flags=zmq.DONTWAIT
                        )
                    except zmq.Again:
                        logger.debug("Gaze publish skipped (no subscriber / HWM reached)")
            except Exception as e:
                logger.warning("Gaze payload build failed: %s", repr(e))







            # --- Guard people count cases consistent with your flow
            if idPos is None:
                continue
            if idPos[1] == -1:
                continue
            if idPos[1] == -2:
                continue

            left_id, right_id = idPos
            if turn_n >= len(turn_order) - 1:
                config_finished = True

            # --- Core decision using DEPTH PLANE @ 30cm (with hysteresis)
            def decide_event_for_person(person, who_key, who_label, vis_img=None):
                """
                Returns (any_extended, any_retracted) for this person.
                Uses per-side checks + hysteresis + tiny debounce to avoid false triggers.
                """
                parts = elbows_shoulders_wrists_px(person, W, H)

                # Draw uses *current* latched state for color
                latched_status = 'EXT' if state[who_key] == 1 else 'RET'

                # Per-side checks
                side_flags = {"Left": {"ext": False, "ret": False},
                              "Right":{"ext": False, "ret": False}}

                for side in ("Left", "Right"):
                    E = parts[side]["E"]
                    Wp = parts[side]["W"]
                    if Wp is None or E is None:
                        continue
                    
                    wx, wy = Wp
                    ex, ey = E

                    wrist_z = median_depth_m(depth, wx, wy, win=3)  # slightly larger window
                    if wrist_z <= 0:
                        if vis_img is not None:
                            draw_status_overlay(vis_img, ex, ey, wx, wy, latched_status, 0.0, 0.0)
                        continue
                    
                    plane_z = plane_reference_z(depth, person, parts, side, W, H)
                    if plane_z is None:
                        if vis_img is not None:
                            draw_status_overlay(vis_img, ex, ey, wx, wy, latched_status, 0.0, wrist_z)
                        continue
                    
                    # Per-side hysteresis decision
                    if wrist_z < (plane_z - PLANE_HYST_M):
                        side_flags[side]["ext"] = True
                    elif wrist_z > (plane_z + PLANE_HYST_M):
                        side_flags[side]["ret"] = True

                    if vis_img is not None:
                        # draw using the *latched* status color for stability
                        draw_status_overlay(vis_img, ex, ey, wx, wy, latched_status, plane_z, wrist_z)

                # Combine both sides
                any_ext = side_flags["Left"]["ext"] or side_flags["Right"]["ext"]
                any_ret = side_flags["Left"]["ret"] or side_flags["Right"]["ret"]

                # Debounce: require 2 consecutive frames
                if who_key == "left":
                    streak["left_ext"] = (streak["left_ext"] + 1) if any_ext else 0
                    streak["left_ret"] = (streak["left_ret"] + 1) if any_ret else 0
                    any_ext = streak["left_ext"] >= STREAK_N
                    any_ret = streak["left_ret"] >= STREAK_N
                else:
                    streak["right_ext"] = (streak["right_ext"] + 1) if any_ext else 0
                    streak["right_ret"] = (streak["right_ret"] + 1) if any_ret else 0
                    any_ext = streak["right_ext"] >= STREAK_N
                    any_ret = streak["right_ret"] >= STREAK_N

                return any_ext, any_ret


            # We’ll draw on the same image you already show (if enabled)
            vis_img = None
            if not args.no_display:
                vis_img = datum.cvOutputData if datum.cvOutputData is not None else frame

            # ----- Apply per-turn order logic with the depth-plane booleans -----
            if turn_order[turn_n] == 'b':
                person = datumProcessed[0].poseKeypoints[left_id]
                any_ext, any_ret = decide_event_for_person(person, "left", "Left person", vis_img)

                if checkStatus == 0:
                    # seek EXTENDED
                    if any_ext:
                        buffer.append(f"Left: Arm Extended")
                        buffer_len += 1
                        checkStatus = 1
                        state["left"] = 1
                        logger.info("Left person: Arm EXTENDED (depth-plane)")
                        print("Left person: Arm Extended")
                else:
                    # seek RETRACTED
                    if any_ret:
                        buffer.append(f"Left: Arm Retracted")
                        buffer_len += 1
                        checkStatus = 0
                        state["left"] = 0
                        turn_n += 1
                        logger.info("Left person: Arm RETRACTED (depth-plane)")
                        print("Left person: Arm Retracted")

            elif turn_order[turn_n] == 'p':
                person = datumProcessed[0].poseKeypoints[right_id]
                any_ext, any_ret = decide_event_for_person(person, "right", "Right person", vis_img)

                if checkStatus == 0:
                    if any_ext:
                        buffer.append(f"Right: Arm Extended")
                        buffer_len += 1
                        checkStatus = 1
                        state["right"] = 1
                        logger.info("Right person: Arm EXTENDED (depth-plane)")
                        print("Right person: Arm Extended")
                else:
                    if any_ret:
                        buffer.append(f"Right: Arm Retracted")
                        buffer_len += 1
                        checkStatus = 0
                        state["right"] = 0
                        turn_n += 1
                        logger.info("Right person: Arm RETRACTED (depth-plane)")
                        print("Right person: Arm Retracted")

            elif turn_order[turn_n] == 'y':
                logger.info("Robot's turn detected")
                turn_n += 1

            # ----- ZMQ reply loop (unchanged) -----
            if buffer_it < buffer_len or config_finished:
                try:
                    message = socket.recv(flags=zmq.NOBLOCK)
                    logger.info(f"ZMQ Message received (non-blocking): {message.decode()}")
                    print("Message received:", message)

                    if message.decode() == "Configuration finished":
                        logger.info("Configuration finished. Waiting for next configuration")
                        print("Configuration finished. Waiting for next configuration…")
                        socket.send_string("Received")
                        break

                    socket.send_string(buffer[buffer_it])
                    logger.info(f"ZMQ Sending: {buffer[buffer_it]}")
                    buffer_it += 1

                except zmq.Again:
                    # no message yet
                    pass

            frame_idx += 1

except Exception as e:
    logger.critical(f"Critical error in openpose_server: {e}", exc_info=True)
    print(e)
    sys.exit(-1)
finally:
    try:
        pipe.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()