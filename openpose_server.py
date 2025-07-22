# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import zmq
from sys import platform
import argparse
import websocket
import av  # PyAV library for decoding H264
import numpy as np
import math

import pathlib
#import torch
#import torch.backends.cudnn as cudnn
#from L2CS_Net.l2cs import select_device, draw_gaze, getArch, Pipeline, render

import logging 
from datetime import datetime


# --- Logging Setup ---
log_filename = f"kinova_log_openpose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = logging.getLogger('openpose')
logger.setLevel(logging.DEBUG)  # or INFO

# Avoid adding multiple handlers if re-running the script
if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)



context = zmq.Context()

# Turn on Server
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
logger.info("OpenPose server started and bound to tcp://*:5555")


def display(datumProcessed, idPos=None):
    datum = datumProcessed[0]
    output = datum.cvOutputData    
    keypoints = datum.poseKeypoints

    if output is None or keypoints is None:
        return False

    output = output.copy()
    frame_height, frame_width = output.shape[:2]

    for i, person in enumerate(keypoints):
        right_shoulder = person[2]
        left_shoulder = person[5]

        rx, ry, rc = right_shoulder
        lx, ly, lc = left_shoulder

        if rx > 0 and ry > 0 and rc > 0.7 and \
           lx > 0 and ly > 0 and lc > 0.7:

            x = int(((rx + lx) / 2) * frame_width)
            y = int(((ry + ly) / 2) * frame_height)
            y = max(20, y)

            if idPos is not None:
                if i == idPos[0]:
                    label = "Left"
                elif i == idPos[1]:
                    label = "Right"
                else:
                    label = f"Untracked {i}"
            else:
                label = f"ID {i}"

            cv2.circle(output, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(output, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(output, f"Cl {lc:.2f}", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(output, f"Cr {rc:.2f}", (x, y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    try:

        # Show OpenPose result locally (optional)
        cv2.imshow("OpenPose Result", output)
        key = cv2.waitKey(1)
        return key == 27
    except Exception as e:
        logger.error(f"Exception during imshow/waitKey: {e}")
        print(f"Exception during imshow/waitKey: {e}")
        return False




def printKeypoints(datums):
    datum = datums[0]
    logger.info(f"Body keypoints: {datum.poseKeypoints}")
    print("Body keypoints: \n" + str(datum.poseKeypoints))

def verifyCheckStart(datums):
    datum = datums[0]
    if datum.poseKeypoints is None:
        return
    for person in datum.poseKeypoints:
        for keypoint in person:
            keypointX = keypoint[0]
            if keypointX != 0 and keypointX <= 0.5:
                #print("Arm check not started")
                return False
    return True



def get_shoulder_width(person):
    left_shoulder = person[5]
    right_shoulder = person[2]
    if left_shoulder[0] == 0 or right_shoulder[0] == 0:
        return None
    return abs(right_shoulder[0] - left_shoulder[0])

'''def checkLeftArmExtended(person, ratio=0.8):
    shoulder_x = person[5][0]
    wrist_x = person[7][0]
    if wrist_x == 0 or shoulder_x == 0:
        return False
    shoulder_width = get_shoulder_width(person)
    if shoulder_width is None:
        return False
    return abs(wrist_x - shoulder_x) > (shoulder_width * ratio)

def checkLeftArmRetracted(person, ratio=0.5):
    shoulder_x = person[5][0]
    wrist_x = person[7][0]
    if wrist_x == 0 or shoulder_x == 0:
        return False
    shoulder_width = get_shoulder_width(person)
    if shoulder_width is None:
        return False
    return abs(wrist_x - shoulder_x) < (shoulder_width * ratio)

def checkRightArmExtended(person, ratio=0.8):
    shoulder_x = person[2][0]
    wrist_x = person[4][0]
    if wrist_x == 0 or shoulder_x == 0:
        return False
    shoulder_width = get_shoulder_width(person)
    if shoulder_width is None:
        return False
    return abs(wrist_x - shoulder_x) > (shoulder_width * ratio)

def checkRightArmRetracted(person, ratio=0.5):
    shoulder_x = person[2][0]
    wrist_x = person[4][0]
    if wrist_x == 0 or shoulder_x == 0:
        return False
    shoulder_width = get_shoulder_width(person)
    if shoulder_width is None:
        return False
    return abs(wrist_x - shoulder_x) < (shoulder_width * ratio)'''

def get_2d_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_vector(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    return (dx / norm, dy / norm)

def get_angle_between(v1, v2):
    if v1 is None or v2 is None:
        return None
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    dot = max(min(dot, 1.0), -1.0)  # clamp to [-1, 1]
    return math.acos(dot) * 180 / math.pi  # degrees

def get_torso_length(person):
    left_shoulder = person[5]
    right_shoulder = person[2]
    left_hip = person[11]
    right_hip = person[8]

    torso_top = None
    if left_shoulder[2] > 0.7 and right_shoulder[2] > 0.7:
        torso_top = [(left_shoulder[0] + right_shoulder[0]) / 2,
                     (left_shoulder[1] + right_shoulder[1]) / 2]
    elif left_shoulder[2] > 0.7:
        torso_top = left_shoulder
    elif right_shoulder[2] > 0.7:
        torso_top = right_shoulder

    torso_bottom = None
    if left_hip[2] > 0.7 and right_hip[2] > 0.7:
        torso_bottom = [(left_hip[0] + right_hip[0]) / 2,
                        (left_hip[1] + right_hip[1]) / 2]
    elif left_hip[2] > 0.7:
        torso_bottom = left_hip
    elif right_hip[2] > 0.7:
        torso_bottom = right_hip

    if torso_top is not None and torso_bottom is not None:
        return get_2d_distance(torso_top, torso_bottom)
    return None

def is_any_arm_extended(person, ratio=0.6, angle_min=0, angle_max=40, label="Person"):
    torso_len = get_torso_length(person)
    if torso_len is None:
        return False

    left_shoulder = person[5]
    right_shoulder = person[2]
    left_wrist = person[7]
    right_wrist = person[4]
    left_hip = person[11]
    right_hip = person[8]

    for side, (shoulder, wrist) in zip(["Left", "Right"], [(left_shoulder, left_wrist), (right_shoulder, right_wrist)]):
        if shoulder[2] > 0.7 and wrist[2] > 0.7:
            dist = get_2d_distance(shoulder, wrist)
            arm_vec = get_vector(shoulder, wrist)

            # Torso vector = shoulder to hip
            if side == "Left" and left_hip[2] > 0.7:
                torso_vec = get_vector(shoulder, left_hip)
            elif side == "Right" and right_hip[2] > 0.7:
                torso_vec = get_vector(shoulder, right_hip)
            else:
                continue

            angle = get_angle_between(arm_vec, torso_vec)

            logger.debug(f"{label} - {side} arm → dist: {dist:.2f}, angle wrt torso: {angle:.2f}, range: {angle_min}–{angle_max}")
            if angle is not None and angle_min <= angle <= angle_max and dist > torso_len * ratio:
                logger.info(f"{label} - {side} arm detected as EXTENDED (placing)")
                return True

    return False



def is_any_arm_retracted(person, ratio=0.5, angle_threshold=120, label="Person"):
    torso_len = get_torso_length(person)
    if torso_len is None:
        return False

    left_shoulder = person[5]
    right_shoulder = person[2]
    left_wrist = person[7]
    right_wrist = person[4]
    left_hip = person[11]
    right_hip = person[8]

    for side, (shoulder, wrist) in zip(["Left", "Right"], [(left_shoulder, left_wrist), (right_shoulder, right_wrist)]):
        if shoulder[2] > 0.7 and wrist[2] > 0.7:
            dist = get_2d_distance(shoulder, wrist)
            arm_vec = get_vector(shoulder, wrist)

            if side == "Left" and left_hip[2] > 0.7:
                torso_vec = get_vector(shoulder, left_hip)
            elif side == "Right" and right_hip[2] > 0.7:
                torso_vec = get_vector(shoulder, right_hip)
            else:
                continue

            angle = get_angle_between(arm_vec, torso_vec)

            logger.debug(f"{label} - {side} arm → dist: {dist:.2f}, angle wrt torso: {angle:.2f}, retracted threshold: {angle_threshold}")
            if angle is not None and angle < angle_threshold and dist < torso_len * ratio:
                logger.info(f"{label} - {side} arm detected as RETRACTED (rest)")
                return True

    return False







def defineIdPos(datums, image_width):
    datum = datums[0]

    if datum.poseKeypoints is None:
        return

    num_people = len(datum.poseKeypoints)
    if num_people == 1:
        return (0, -1)
    if num_people > 2:
        return (0, -2)

    # OpenPose keypoint indices
    RIGHT_SHOULDER = 2
    LEFT_SHOULDER = 5

    def get_avg_shoulder_x(person):
        rs = person[RIGHT_SHOULDER]
        ls = person[LEFT_SHOULDER]

        rs_valid = rs[2] > 0.7
        ls_valid = ls[2] > 0.7

        coords = []
        if rs_valid:
            coords.append(rs[0] * image_width)
        if ls_valid:
            coords.append(ls[0] * image_width)

        if coords:
            return sum(coords) / len(coords)
        else:
            return None

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



def connect_kinova_camera(ws_url):
    ws = websocket.create_connection(ws_url)
    container = av.open(ws.makefile('rb'), format='h264')
    return container

def get_next_frame(cap):
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        return None



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


    # Flags
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--no-display", action="store_true", help="Disable display.")
    #parser.add_argument('--device', help='Device to run model: cpu or gpu:0', default="cpu", type=str)
    #parser.add_argument('--snapshot', help='Path of model snapshot.',
    #                    default='/home/ines/Desktop/Harmony/L2CS_Net/models/student_combined_epoch_148.pkl', type=str)
    #parser.add_argument('--arch', help='Network architecture',
    #                    default='ResNet50', type=str)
    #args, unknown = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/home/ines/Desktop/Harmony/openpose/models"
    params["net_resolution"] = "192x144"
    params["keypoint_scale"] = "3"
    params["camera"] = -1

    

    #for i in range(len(unknown)):
    #    curr_item = unknown[i]
    #    next_item = unknown[i + 1] if i + 1 < len(unknown) else "1"
    #    if "--" in curr_item and "--" in next_item:
    #        key = curr_item.replace('-', '')
    #        if key not in params:
    #            params[key] = "1"
    #    elif "--" in curr_item and "--" not in next_item:
    #        key = curr_item.replace('-', '')
    #        if key not in params:
    #            params[key] = next_item



    # Starting OpenPose
    logger.info("Starting OpenPose")
    print(" Starting OpenPose...")
    opWrapper = op.WrapperPython(op.ThreadManagerMode.Asynchronous)
    opWrapper.configure(params)
    opWrapper.start()

    CWD = pathlib.Path.cwd()
    #cudnn.enabled = True

    #gaze_pipeline = Pipeline(
    #    weights=CWD / 'L2CS_Net' /'models' / 'student_combined_epoch_148.pkl',
    #    arch=args.arch,
    #    device=select_device(args.device, batch_size=1)
    #)
    

    cap = cv2.VideoCapture("/dev/video4")
    if not cap.isOpened():
        logger.error("Failed to open camera /dev/video4")
        print("Failed to open camera /dev/video4")
        sys.exit(1)
    else:
        logger.info("Camera opened successfully")
        print("Camera opened")

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


        while not userWantsToExit:
            # Get frame from Kinova camera
            frame = get_next_frame(cap)
            if frame is None:
                logger.warning("Failed to get frame from camera.")
                continue

            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            datumProcessed = [datum]  # wrap in list for compatibility

            # Get gaze estimation results from L2CS
            #gaze_results = gaze_pipeline.step(frame)

            # Overlay gaze vectors on a copy of the original frame
            #gaze_frame = render(frame.copy(), gaze_results)


            # Print number of people detected if it changes
            if datum.poseKeypoints is not None:
                current_count = len(datum.poseKeypoints)
            else:
                current_count = 0
            
            if current_count != last_detected_count:
                #logger.info(f"Number of people detected changed: {current_count}") 
                last_detected_count = current_count

            frame_height, frame_width = frame.shape[:2]

             # Analyze keypoints
            idPos = defineIdPos(datumProcessed, frame_width)

            #if not args.no_display:
            '''try:
                # Get OpenPose output
                frame_with_openpose = datum.cvOutputData
                # Blend OpenPose + Gaze output for visualization
                #combined_display = cv2.addWeighted(frame_with_openpose, 0.6, gaze_frame, 0.4, 0)
                # Show the result
                cv2.imshow("OpenPose", frame_with_openpose) #combined_display)
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    userWantsToExit = True
            except Exception as e:
                logger.error(f"Display exception: {e}")
                print(f"Display exception: {e}")'''
            
            try:
                if display(datumProcessed, idPos):
                    userWantsToExit = True
            except Exception as e:
                logger.error(f"Display exception: {e}")
                print(f"Display exception: {e}")



            if idPos is None:
                 #print("0 people detected")
                continue
             
            if idPos[1] == -1:
                 #print("Only 1 person detected")
                continue
             
            if idPos[1] == -2:
                 #print("More than 2 people detected")
                continue


            left_id = idPos[0]
            right_id = idPos[1]

            if turn_n >= len(turn_order) - 1:
                config_finished = True

            '''elif turn_order[turn_n] == 'b':
                person = datumProcessed[0].poseKeypoints[left_id]  # <-- move here
                if checkStatus == 0:
                    if checkLeftArmExtended(person):
                        action_flag = True
                        buffer.append("Left: Arm Extended")
                        buffer_len += 1
                        checkStatus = 1
                        logger.info("Left Person: Arm Extended detected")
                        print("Left: Arm Extended")
                elif checkStatus == 1:
                    if checkLeftArmRetracted(person):
                        action_flag = True
                        buffer.append("Left: Arm Retracted")
                        buffer_len += 1
                        checkStatus = 0
                        turn_n += 1
                        logger.info("Left Person: Arm Retracted detected")
                        print("Left: Arm Retracted")

            elif turn_order[turn_n] == 'p':
                person = datumProcessed[0].poseKeypoints[right_id]  # <-- move here
                if checkStatus == 0:
                    if checkRightArmExtended(person):
                        action_flag = True
                        buffer.append("Right: Arm Extended")
                        buffer_len += 1
                        checkStatus = 1
                        logger.info("Right Person: Arm Extended detected")
                        print("Right: Arm Extended")
                elif checkStatus == 1:
                    if checkRightArmRetracted(person):
                        action_flag = True
                        buffer.append("Right: Arm Retracted")
                        buffer_len += 1
                        checkStatus = 0
                        turn_n += 1
                        logger.info("Right Person: Arm Retracted detected")
                        print("Right: Arm Retracted")'''
            if turn_order[turn_n] == 'b':
                person = datumProcessed[0].poseKeypoints[left_id]
                if checkStatus == 0:
                    if is_any_arm_extended(person):
                        action_flag = True
                        buffer.append("Left: Arm Extended")
                        buffer_len += 1
                        checkStatus = 1
                        logger.info("Left Person: Arm Extended detected")
                        print("Left: Arm Extended")
                elif checkStatus == 1:
                    if is_any_arm_retracted(person):
                        action_flag = True
                        buffer.append("Left: Arm Retracted")
                        buffer_len += 1
                        checkStatus = 0
                        turn_n += 1
                        logger.info("Left Person: Arm Retracted detected")
                        print("Left: Arm Retracted")

            elif turn_order[turn_n] == 'p':
                person = datumProcessed[0].poseKeypoints[right_id]
                if checkStatus == 0:
                    if is_any_arm_extended(person):
                        action_flag = True
                        buffer.append("Right: Arm Extended")
                        buffer_len += 1
                        checkStatus = 1
                        logger.info("Right Person: Arm Extended detected")
                        print("Right: Arm Extended")
                elif checkStatus == 1:
                    if is_any_arm_retracted(person):
                        action_flag = True
                        buffer.append("Right: Arm Retracted")
                        buffer_len += 1
                        checkStatus = 0
                        turn_n += 1
                        logger.info("Right Person: Arm Retracted detected")
                        print("Right: Arm Retracted")

            elif turn_order[turn_n] == 'y':
                logger.info("Robot's turn detected")
                turn_n += 1



             #ZMQ communication
            if buffer_it < buffer_len or config_finished:
                try:
                    message = socket.recv(flags=zmq.NOBLOCK)
                    logger.info(f"ZMQ Message received (non-blocking): {message.decode()}")
                    print("Message received:", message)
  
                    if message.decode() == "Configuration finished":
                        logger.info("Configuration finished. Waiting for next configuration")
                        print("Configuration finished. Waiting for next configuration...")
                        socket.send_string("Received")
                        break
  
                    socket.send_string(buffer[buffer_it])
                    logger.info(f"ZMQ Sending: {buffer[buffer_it]}")
                    buffer_it += 1
                    action_flag = False
  
                except zmq.Again:
                    #logger.debug("No ZMQ message received yet")
                    pass

        
except Exception as e:
    logger.critical(f"Critical error in openpose_server: {e}", exc_info=True)
    print(e)
    sys.exit(-1)