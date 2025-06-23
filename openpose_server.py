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

context = zmq.Context()

# Turn on Server
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

#def display(datums):
#    datum = datums[0]
#    cv2.imshow("OpenPose 1.7.0 - Python API", datum.cvOutputData)
#    key = cv2.waitKey(1)
#    return (key == 27)

def display(datumProcessed):
   
    output = datumProcessed[0].cvOutputData
    if output is None:
        print("❌ cvOutputData is None")
        return False

    try:
        cv2.imshow("OpenPose Result", output)
        key = cv2.waitKey(1)
        return key == 27
    except Exception as e:
        print(f"❌ Exception during imshow/waitKey: {e}")
        return False



def printKeypoints(datums):
    datum = datums[0]
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

def is_valid_person(person, min_height=50, max_height=300,
                    min_arm=30, max_arm=300,
                    min_shoulder_width=30, max_shoulder_width=300):
    # Extract keypoints
    neck = person[1]
    mid_hip = person[8]
    left_shoulder = person[5]
    right_shoulder = person[2]
    left_wrist = person[7]
    right_wrist = person[4]

    # Torso height (neck to mid-hip)
    if np.any(neck[:2] == 0) or np.any(mid_hip[:2] == 0):
        return False
    torso_height = np.linalg.norm(neck[:2] - mid_hip[:2])
    if not (min_height < torso_height < max_height):
        return False

    # Left arm
    if np.any(left_shoulder[:2] == 0) or np.any(left_wrist[:2] == 0):
        return False
    left_arm_len = np.linalg.norm(left_shoulder[:2] - left_wrist[:2])
    if not (min_arm < left_arm_len < max_arm):
        return False

    # Right arm
    if np.any(right_shoulder[:2] == 0) or np.any(right_wrist[:2] == 0):
        return False
    right_arm_len = np.linalg.norm(right_shoulder[:2] - right_wrist[:2])
    if not (min_arm < right_arm_len < max_arm):
        return False

    # Shoulder width
    if np.any(left_shoulder[:2] == 0) or np.any(right_shoulder[:2] == 0):
        return False
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    if not (min_shoulder_width < shoulder_width < max_shoulder_width):
        return False

    return True


def get_first_visible_x(person):
    for keypoint in person:
        if keypoint[0] != 0:
            return keypoint[0]
    return float('inf')


def get_shoulder_width(person):
    left_shoulder = person[5]
    right_shoulder = person[2]
    if left_shoulder[0] == 0 or right_shoulder[0] == 0:
        return None
    return abs(right_shoulder[0] - left_shoulder[0])

def checkLeftArmExtended(person, ratio=0.8):
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
    return abs(wrist_x - shoulder_x) < (shoulder_width * ratio)


'''def get_shoulder_width(person):
    left_shoulder = person[5]
    right_shoulder = person[2]
    if left_shoulder[0] == 0 or right_shoulder[0] == 0:
        return None
    return abs(right_shoulder[0] - left_shoulder[0])

def checkArmExtendedLeft(datums, left_id, ratio=0.8):
    datum = datums[0]
    if datum.poseKeypoints is None:
        return False
    person = datum.poseKeypoints[left_id]

    shoulder_x = person[5][0]
    wrist_x = person[7][0]
    if wrist_x == 0 or shoulder_x == 0:
        return False

    shoulder_width = get_shoulder_width(person)
    if shoulder_width is None:
        return False

    return abs(wrist_x - shoulder_x) > (shoulder_width * ratio)

def checkArmRetractedLeft(datums, left_id, ratio=0.5):
    datum = datums[0]
    if datum.poseKeypoints is None:
        return False
    person = datum.poseKeypoints[left_id]

    shoulder_x = person[5][0]
    wrist_x = person[7][0]
    if wrist_x == 0 or shoulder_x == 0:
        return False

    shoulder_width = get_shoulder_width(person)
    if shoulder_width is None:
        return False

    return abs(wrist_x - shoulder_x) < (shoulder_width * ratio)

def checkArmExtendedRight(datums, right_id, ratio=0.8):
    datum = datums[0]
    if datum.poseKeypoints is None:
        return False
    person = datum.poseKeypoints[right_id]

    shoulder_x = person[2][0]
    wrist_x = person[4][0]
    if wrist_x == 0 or shoulder_x == 0:
        return False

    shoulder_width = get_shoulder_width(person)
    if shoulder_width is None:
        return False

    return abs(wrist_x - shoulder_x) > (shoulder_width * ratio)

def checkArmRetractedRight(datums, right_id, ratio=0.5):
    datum = datums[0]
    if datum.poseKeypoints is None:
        return False
    person = datum.poseKeypoints[right_id]

    shoulder_x = person[2][0]
    wrist_x = person[4][0]
    if wrist_x == 0 or shoulder_x == 0:
        return False

    shoulder_width = get_shoulder_width(person)
    if shoulder_width is None:
        return False

    return abs(wrist_x - shoulder_x) < (shoulder_width * ratio)
'''

'''def checkArmExtendedLeft(datums, left_id):
    datum = datums[0]
    if datum.poseKeypoints is None:
        return
    person = datum.poseKeypoints[left_id]
    for keypoint in person:
        keypointX = keypoint[0]
        if keypointX != 0 and keypointX >= 0.40:
            return True
    return False

def checkArmRetractedLeft(datums, left_id):
    datum = datums[0]
    if datum.poseKeypoints is None:
        return
    person = datum.poseKeypoints[left_id]
    for keypoint in person:
        keypointX = keypoint[0]
        if keypointX != 0 and keypointX >= 0.40:
            return False
    return True

def checkArmExtendedRight(datums, right_id):
    datum = datums[0]
    if datum.poseKeypoints is None:
        return
    person = datum.poseKeypoints[right_id]
    for keypoint in person:
        keypointX = keypoint[0]
        if keypointX != 0 and keypointX <= 0.60:
            return True
    return False

def checkArmRetractedRight(datums, right_id):
    datum = datums[0]
    if datum.poseKeypoints is None:
        return
    person = datum.poseKeypoints[right_id]
    for keypoint in person:
        keypointX = keypoint[0]
        if keypointX != 0 and keypointX <= 0.60:
            return False
    return True'''


#def defineIdPos(datums):
#    datum = datums[0]
#    keypoints = datum.poseKeypoints
#    if keypoints is None:
#        return
#
#    valid_people = []
#    for idx, person in enumerate(keypoints):
#        if is_valid_person(person):
#            valid_people.append((idx, person))
#
#    if len(valid_people) == 1:
#        return (valid_people[0][0], -1)
#    if len(valid_people) < 1:
#        return
#    if len(valid_people) > 2:
#        return (0, -2)
#
#    # Exactly 2 valid people — now assign left/right
#    idx0, p0 = valid_people[0]
#    idx1, p1 = valid_people[1]
#
#
#    p0X = get_first_visible_x(p0)
#    p1X = get_first_visible_x(p1)
#
#    if p0X < p1X:
#        return (idx0, idx1)
#    else:
#        return (idx1, idx0)
#
def defineIdPos(datums): 
    datum = datums[0]
    if datum.poseKeypoints is None:
        return
    if len(datum.poseKeypoints) == 1:
        return (0,-1)
    if len(datum.poseKeypoints) > 2:
        return (0,-2)
    p0 = datum.poseKeypoints[1]
    p1 = datum.poseKeypoints[0]
    for keypointP0 in p0:
        if keypointP0[0] != 0:
            p0X = keypointP0[0]
            break
    for keypointP1 in p1:
        if keypointP1[0] != 0:
            p1X = keypointP1[0]
            break
    if p0X < p1X:
        return (0, 1)
    else:
        return (1, 0)

#def defineIdPos(datums):
#    datum = datums[0]
#    if datum.poseKeypoints is None:
#        return
#    if len(datum.poseKeypoints) == 1:
#        return (0, 0)  # pretend person 0 is both left and right
#    if len(datum.poseKeypoints) > 2:
#        return (0, -2)  # just pick first two
#    p0 = datum.poseKeypoints[0]
#    p1 = datum.poseKeypoints[1]
#    for keypointP0 in p0:
#        if keypointP0[0] != 0:
#            p0X = keypointP0[0]
#            break
#    for keypointP1 in p1:
#        if keypointP1[0] != 0:
#            p1X = keypointP1[0]
#            break
#    if p0X < p1X:
#        return (0, 1)
#    else:
#        return (1, 0)


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
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e


    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", action="store_true", help="Disable display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/home/ines/Desktop/Harmony/openpose/models"
    params["net_resolution"] = "192x144"
    params["keypoint_scale"] = "3"
    params["camera"] = -1
    


    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    print(" Starting OpenPose...")
    opWrapper = op.WrapperPython(op.ThreadManagerMode.Asynchronous)
    opWrapper.configure(params)
    opWrapper.start()
    

    cap = cv2.VideoCapture("/dev/video4")
    if not cap.isOpened():
        print("❌ Failed to open camera /dev/video4")
        sys.exit(1)
    else:
        print("Camera opened")

    while True:

        print(" OpenPose initialized — waiting for turn_order...")
        turn_order_aux = socket.recv()
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
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            datumProcessed = [datum]  # wrap in list for compatibility


            # Print number of people detected if it changes
            if datum.poseKeypoints is not None:
                current_count = len(datum.poseKeypoints)
            else:
                current_count = 0
            
            if current_count != last_detected_count:
                print(f"👥 Detected people: {current_count}")
                last_detected_count = current_count

            if not args[0].no_display:
                 # Display image
                 userWantsToExit = display(datumProcessed)

             # Analyze keypoints
            idPos = defineIdPos(datumProcessed)

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

            elif turn_order[turn_n] == 'b':
                person = datumProcessed[0].poseKeypoints[left_id]  # <-- move here
                if checkStatus == 0:
                    if checkLeftArmExtended(person):
                        action_flag = True
                        buffer.append("Left: Arm Extended")
                        buffer_len += 1
                        checkStatus = 1
                        print("Left: Arm Extended")
                elif checkStatus == 1:
                    if checkLeftArmRetracted(person):
                        action_flag = True
                        buffer.append("Left: Arm Retracted")
                        buffer_len += 1
                        checkStatus = 0
                        turn_n += 1
                        print("Left: Arm Retracted")

            elif turn_order[turn_n] == 'p':
                person = datumProcessed[0].poseKeypoints[right_id]  # <-- move here
                if checkStatus == 0:
                    if checkRightArmExtended(person):
                        action_flag = True
                        buffer.append("Right: Arm Extended")
                        buffer_len += 1
                        checkStatus = 1
                        print("Right: Arm Extended")
                elif checkStatus == 1:
                    if checkRightArmRetracted(person):
                        action_flag = True
                        buffer.append("Right: Arm Retracted")
                        buffer_len += 1
                        checkStatus = 0
                        turn_n += 1
                        print("Right: Arm Retracted")


            elif turn_order[turn_n] == 'y':
                turn_n += 1



             #ZMQ communication
            if buffer_it < buffer_len or config_finished:
                try:
                    message = socket.recv(flags=zmq.NOBLOCK)
                    print("Message received:", message)
#  
                    if message.decode() == "Configuration finished":
                        print("Configuration finished. Waiting for next configuration...")
                        socket.send_string("Received")
                        break
#  
                    socket.send_string(buffer[buffer_it])
                    buffer_it += 1
                    action_flag = False
#  
                except zmq.Again:
                    pass

        
except Exception as e:
    print(e)
    sys.exit(-1)