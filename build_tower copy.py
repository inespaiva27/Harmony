#!/usr/bin/env python3
###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import os
import sys
import rospy
import time
import zmq
import logging 
import json
from datetime import datetime 
from kortex_driver.srv import *
from kortex_driver.msg import *

# --- Shared timestamped log directory ---
if "RUN_TIMESTAMP" not in os.environ:
    os.environ["RUN_TIMESTAMP"] = datetime.now().strftime("%Y%m%d_%H%M")

timestamp = os.environ["RUN_TIMESTAMP"]
log_dir = os.path.join(os.path.dirname(__file__), "logs", timestamp)
os.makedirs(log_dir, exist_ok=True)

# --- Logging setup ---
log_filename = os.path.join(log_dir, f"kinova_log_buildtower_{timestamp}.log")
logger = logging.getLogger('BuildTower')
logger.setLevel(logging.DEBUG)  # or INFO

if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# --- Gaze-only logging (port 5556) ---
gaze_log_filename = os.path.join(log_dir, f"gaze_5556_{timestamp}.log")
gaze_logger = logging.getLogger('Gaze5556')
gaze_logger.setLevel(logging.INFO)

# Avoid duplicate handlers if script is reloaded
if not gaze_logger.hasHandlers():
    gaze_file_handler = logging.FileHandler(gaze_log_filename)
    gaze_formatter = logging.Formatter('%(asctime)s - %(message)s')
    gaze_file_handler.setFormatter(gaze_formatter)
    gaze_logger.addHandler(gaze_file_handler)



context = zmq.Context()

# Create socket outside the try loop
socket = context.socket(zmq.REQ)

# Try to connect to OpenPose server
connected = False
for i in range(10):
    try:
        socket.connect("tcp://localhost:5555")
        logger.info("Connected to OpenPose server")
        print("Connected to OpenPose server")

        sub = context.socket(zmq.SUB)
        sub.setsockopt(zmq.RCVHWM, 10)     # avoid big buffers
        sub.setsockopt(zmq.CONFLATE, 1)    # keep only the latest gaze message
        sub.connect("tcp://localhost:5556")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        logger.info("Connected to Gaze server")

        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        poller.register(socket, zmq.POLLIN)
        
        connected = True


        break
    except Exception as e:
        logger.warning(f"Connection attempt {i+1} failed: {e}")
        print(f"Connection attempt {i+1} failed: {e}")
        time.sleep(1)

if not connected:
    logger.critical("Failed to connect to OpenPose ZMQ server. Exiting.")
    print("Failed to connect to OpenPose ZMQ server. Exiting.")
    sys.exit(1)

block_1_start = ConstrainedPose()
block_1_target = ConstrainedPose()
block_2_start = ConstrainedPose()
block_2_target = ConstrainedPose()
block_3_start = ConstrainedPose()
block_3_target = ConstrainedPose()

class BuildTower:
    def __init__(self):
        try:
            rospy.init_node('build_tower_python')

            self.HOME_ACTION_IDENTIFIER = 2
            self.ZERO_ACTION_IDENTIFIER = 4

            self.action_topic_sub = None
            self.all_notifs_succeeded = True

            self.all_notifs_succeeded = True

            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 7)
            self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

            rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))
            logger.info(f"Robot parameters: robot_name={self.robot_name}, dof={self.degrees_of_freedom}, gripper_present={self.is_gripper_present}")

            # Init the action topic subscriber
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            # Init the services
            clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)
            logger.info(f"Service initialized: {clear_faults_full_name}")

            read_action_full_name = '/' + self.robot_name + '/base/read_action'
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)
            logger.info(f"Service initialized: {read_action_full_name}")

            execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)
            logger.info(f"Service initialized: {execute_action_full_name}")

            set_cartesian_reference_frame_full_name = '/' + self.robot_name + '/control_config/set_cartesian_reference_frame'
            rospy.wait_for_service(set_cartesian_reference_frame_full_name)
            self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name, SetCartesianReferenceFrame)
            logger.info(f"Service initialized: {set_cartesian_reference_frame_full_name}")

            activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)
            logger.info(f"Service initialized: {activate_publishing_of_action_notification_full_name}")

            send_gripper_command_full_name = '/' + self.robot_name + '/base/send_gripper_command'
            rospy.wait_for_service(send_gripper_command_full_name)
            self.send_gripper_service = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)
            logger.info(f"Service initialized: {send_gripper_command_full_name}")
            #self.send_gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

            self.yaw_bias_deg = 0.0
            self.pitch_bias_deg = 0.0

            # For throttling gaze updates (only every 3 s)
            self.last_gaze_update_time = 0.0



        except Exception as e:
            self.is_init_success = False
            logger.error(f"Initialization failed: {e}", exc_info=True)

        else:
            self.is_init_success = True
            logger.info("BuildTower class initialized successfully.")

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                logger.info("Robot action ended.")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                logger.warning("Robot action aborted.")
                self.all_notifs_succeeded = False
                return False
            else:
                time.sleep(0.01)

    def robot_clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call ClearFaults")
            logger.error(f"Failed to call ClearFaults: {e}")
            return False

        else:
            rospy.loginfo("Cleared the faults successfully")
            logger.info("Robot faults cleared successfully.")
            rospy.sleep(2.5)
            return True

    def home_the_robot(self):
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        self.last_action_notif_type = None
        try:
            res = self.read_action(req)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call ReadAction")
            logger.error(f"Failed to call ReadAction for home: {e}")
            return False

        # Execute the HOME action if we could read it
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot home...")
            logger.info("Sending robot to home position.")
            try:
                self.execute_action(req)
            except rospy.ServiceException as e:
                rospy.logerr("Failed to call ExecuteAction")
                logger.error(f"Failed to call ExecuteAction: {e}")
                return False
            else:
                time.sleep(0.6)
                return 1
            
    def zero_the_robot(self):
      # The Zero Action is used to zero the robot. It cannot be deleted and is always ID #4:
      req = ReadActionRequest()
      req.input.identifier = self.ZERO_ACTION_IDENTIFIER
      self.last_action_notif_type = None
      try:
          res = self.read_action(req)
      except rospy.ServiceException as e:
          rospy.logerr("Failed to call ReadAction")
          logger.error(f"Failed to call ReadAction for zero: {e}")
          return False
      # Execute the ZERO action if we could read it
      else:
          # What we just read is the input of the ExecuteAction service
          req = ExecuteActionRequest()
          req.input = res.output
          rospy.loginfo("Sending the robot zero...")
          logger.info("Sending robot to zero position.")
          try:
              self.execute_action(req)
          except rospy.ServiceException as e:
              rospy.logerr("Failed to call ExecuteAction")
              logger.error(f"Failed to call ExecuteAction for zero: {e}")
              return False
          else:
              time.sleep(0.6)
              return 1
          

    def robot_set_cartesian_reference_frame(self):
        # Prepare the request with the frame we want to set
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED

        # Call the service
        try:
            self.set_cartesian_reference_frame()
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call ExecuteAction")
            logger.error(f"Failed to call ExecuteAction: {e}")
            return False
        else:
            rospy.loginfo("Set the cartesian reference frame successfully")
            logger.info("Cartesian reference frame set successfully.")
            return True

        # Wait a bit
        rospy.sleep(0.25)

    def subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        logger.info("Activating action notifications.")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            logger.error(f"Failed to call OnNotificationActionTopic: {e}")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")
            logger.info("Action Notifications successfully activated.")

        rospy.sleep(1.0)

        return True
        

    def send_gripper_command(self, value):
        # Initialize the request
        # Close the gripper
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo("Sending the gripper command...")
        logger.info(f"Sending gripper command: value={value}")

        # Call the service 
        try:
            #self.send_gripper_command(req)
            self.send_gripper_service(req)

        except rospy.ServiceException as e:
            rospy.logerr("Failed to call SendGripperCommand")
            logger.error(f"Failed to call SendGripperCommand: {e}")
            return False
        else:
            time.sleep(0.5)
            logger.info(f"Gripper command {value} sent successfully.")
            return True
        
    def parse_information(self, argv_n):
        logger.info(f"Parsing information from file: {sys.argv[argv_n]}")
        try:
            f = open(sys.argv[argv_n], 'r')
        except FileNotFoundError:
            logger.error(f"File not found: {sys.argv[argv_n]}")
            print(f"File not found: {sys.argv[argv_n]}")
            sys.exit(1)


        # y = robot; p = left; b = blue
        turn_order = f.readline()

        # 1 = square; 2 = rectangle; 3 = semicircle; 4 = bridge
        object_order = f.readline()

        logger.info(f"Parsed turn_order: {turn_order}")
        logger.info(f"Parsed object_order: {object_order}")

        # Helper function to read pose
        def read_pose(file_handle, pose_obj, block_name):
            line = file_handle.readline().strip()
            if line == block_name:
                pose_obj.target_pose.x = float(file_handle.readline())
                pose_obj.target_pose.y = float(file_handle.readline())
                pose_obj.target_pose.z = float(file_handle.readline())
                pose_obj.target_pose.theta_x = float(file_handle.readline())
                pose_obj.target_pose.theta_y = float(file_handle.readline())
                pose_obj.target_pose.theta_z = float(file_handle.readline())
                return True
            return False

        read_pose(f, block_1_start, "B1 start")
        read_pose(f, block_1_target, "B1 target")
        read_pose(f, block_2_start, "B2 start")
        read_pose(f, block_2_target, "B2 target")
        read_pose(f, block_3_start, "B3 start")
        read_pose(f, block_3_target, "B3 target")

        return (turn_order, object_order)
    
    
    def set_speed(self, cons_pose, translation, orientation):
        my_cartesian_speed = CartesianSpeed()
        my_cartesian_speed.translation = translation # m/s
        my_cartesian_speed.orientation = orientation # deg/s
        cons_pose.constraint.oneof_type.speed.append(my_cartesian_speed)
        logger.debug(f"Set speed for pose: translation={translation}, orientation={orientation}")

    def go_to_position(self, pos):

        logger.info(f"Attempting to go to position: x={pos.target_pose.x}, y={pos.target_pose.y}, z={pos.target_pose.z}, theta_x={pos.target_pose.theta_x}, theta_y={pos.target_pose.theta_y}, theta_z={pos.target_pose.theta_z}")
        self.set_speed(pos, 1, 100)

        req = ExecuteActionRequest()
        req.input.oneof_action_parameters.reach_pose.append(pos)
        req.input.name = "pose"
        req.input.handle.action_type = ActionType.REACH_POSE

        rospy.loginfo("Sending pose...")
        self.last_action_notif_type = None
        try:
            self.execute_action(req)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to send pose")
            logger.error(f"Failed to send pose: {e}")
            return False
        else:
            rospy.loginfo("Waiting for pose to finish...")
            logger.info("Pose action initiated, waiting for completion.")

        self.wait_for_action_end_or_abort()

        return True

    def control_gripper(self, value): # 0 is open, 1 is fully closed
        if self.is_gripper_present:
            self.send_gripper_command(value)
            rospy.loginfo("Moving gripper...")
            return True
        else:
            rospy.logwarn("No gripper is present on the arm.")
            logger.warning("No gripper present, cannot control.")
            return False
    

    def move_block(self, pos1, pos2, turn_n, block_shape):

        logger.info(f"Moving block from start {pos1.target_pose.z} to target {pos2.target_pose.z}. Turn: {turn_n}, Block Shape: {block_shape}")

        self.control_gripper(0)

        pos1.target_pose.z += 0.045
        self.go_to_position(pos1)

        pos1.target_pose.z -= 0.045
        self.go_to_position(pos1)

        if block_shape == 1:
            self.control_gripper(0.075)
        elif block_shape == 2:
            self.control_gripper(0.7)
        elif block_shape == 3:
            self.control_gripper(0.2)
        elif block_shape == 4:
            self.control_gripper(0.7)

        pos1.target_pose.z += 0.045 + turn_n*0.045
        self.go_to_position(pos1)

        pos2.target_pose.z += 0.045
        self.go_to_position(pos2)

        pos2.target_pose.z -= 0.045
        self.go_to_position(pos2)

        self.control_gripper(0)

        pos2.target_pose.z += 0.045
        self.go_to_position(pos2)


        return True


    def look_at_tower(self):
        tower_pos = ConstrainedPose()
        tower_pos.target_pose.x = 0.50
        tower_pos.target_pose.y = 0
        tower_pos.target_pose.z = 0.30
        tower_pos.target_pose.theta_x = 110
        tower_pos.target_pose.theta_y = 0
        tower_pos.target_pose.theta_z = 90
        # Look at tower
        logger.info("Robot moving to look at tower.")
        self.go_to_position(tower_pos)


    def look_at_person_1(self):
        person_1_pos = ConstrainedPose()
        person_1_pos.target_pose.x = 0.5
        person_1_pos.target_pose.y = 0.08
        person_1_pos.target_pose.z = 0.50
        person_1_pos.target_pose.theta_x = 80
        person_1_pos.target_pose.theta_y = 0
        person_1_pos.target_pose.theta_z = 105
        # Look at person 1
        logger.info("Robot moving to look at person in the right.")
        self.go_to_position(person_1_pos)
    

    def look_at_person_2(self):
        person_2_pos = ConstrainedPose()
        person_2_pos.target_pose.x = 0.5
        person_2_pos.target_pose.y = -0.08
        person_2_pos.target_pose.z = 0.50
        person_2_pos.target_pose.theta_x = 80
        person_2_pos.target_pose.theta_y = 0
        person_2_pos.target_pose.theta_z = 75
        # Look at person 2
        logger.info("Robot moving to look at person in the left.")
        self.go_to_position(person_2_pos)

    def look_at_ground(self):
        ground_pose = ConstrainedPose()
        ground_pose.target_pose.x = 0.3
        ground_pose.target_pose.y = 0
        ground_pose.target_pose.z = 0.1
        ground_pose.target_pose.theta_x = 180
        ground_pose.target_pose.theta_y = 0
        ground_pose.target_pose.theta_z = 90
        # Look at ground
        logger.info("Robot moving to look at ground.")
        self.go_to_position(ground_pose)


    def _decide_target_from_data(self, data, yaw_thr=15.0, pitch_thr=15.0):
        """
        Stateless mutual-gaze decision from a SINGLE gaze snapshot.

        Returns:
          "left", "right", or None
        """
        best_id = None
        best_yaw_abs = 180.0

        people = data.get("people", [])
        for e in people:
            pid = e.get("id")          # "left" / "right"
            raw_yaw   = e.get("yaw_deg")
            raw_pitch = e.get("pitch_deg")
            if pid not in ("left", "right"):
                continue
            if raw_yaw is None or raw_pitch is None:
                continue

            # camera -> robot
            yaw   = float(raw_yaw)   - self.yaw_bias_deg
            pitch = float(raw_pitch) - self.pitch_bias_deg

            if abs(yaw) <= yaw_thr and abs(pitch) <= pitch_thr:
                if abs(yaw) < best_yaw_abs:
                    best_yaw_abs = abs(yaw)
                    best_id = pid

        return best_id

    def _look_once_from_payload(self, payload):
        """
        Decode ONE payload and immediately move head, without storing any state.
        """
        try:
            data = json.loads(payload)
        except Exception:
            return

        target = self._decide_target_from_data(data)

        if target == "right":
            self.look_at_person_1()
        elif target == "left":
            self.look_at_person_2()
        else:
            self.look_at_tower()


    def wait_openpose_reply_with_gaze(self, request_str, gaze_enabled):
        """
        Send a request to the OpenPose server on the REQ socket and wait for reply.

        If gaze_enabled is True:
          - while waiting for the reply, read gaze from SUB
          - move the head via look_at_mutual_gazer()

        If gaze_enabled is False:
          - just do a simple blocking send/recv (no tracking)
        """
        # --- No gaze condition → old behavior ---
        if not gaze_enabled:
            logger.info("Sending to OpenPose (no gaze tracking): %s", request_str)
            print(f"Sending request: {request_str}")
            socket.send_string(request_str)
            msg = socket.recv()
            logger.info("Received reply from OpenPose: %s", msg.decode())
            print(f"Received reply [ {msg} ]")
            return msg

        # --- Gaze-enabled condition ---
        logger.info("Sending to OpenPose with gaze tracking: %s", request_str)
        print(f"Sending request (with gaze tracking): {request_str}")
        socket.send_string(request_str)

        while True:
            socks = dict(poller.poll(timeout=50))  # 50 ms

            # 1) PRIORITY: reply from OpenPose?
            if socket in socks and socks[socket] & zmq.POLLIN:
                msg = socket.recv()
                logger.info(
                    "Received reply from OpenPose (with gaze tracking): %s",
                    msg.decode()
                )
                print(f"Received reply [ {msg} ]")
                return msg

            # 2) Gaze data (only if no pending reply)
            if sub in socks and socks[sub] & zmq.POLLIN:
                now = time.time()

                # Only CHECK every 3 seconds, using the CURRENT packet only
                if now - self.last_gaze_update_time >= 3.0:
                    payload = self._recv_gaze_packet(sub, flags=zmq.NOBLOCK)
                    if payload is not None:
                        # Optional: log, but don't store anything important
                        gaze_logger.info("Snapshot gaze used for decision: %s", payload.decode('utf-8', errors='ignore'))
                        self._look_once_from_payload(payload)
                    self.last_gaze_update_time = now
                else:
                    # Just discard to keep buffer fresh (no decision)
                    _ = self._recv_gaze_packet(sub, flags=zmq.NOBLOCK)




    def send_turn_order(self, turn_order):
        logger.info(f"Sending turn order to OpenPose server: {turn_order}")
        print("Sending turn order...")

        self.drain_gaze(sub)
        socket.send_string(turn_order)
        self.drain_gaze(sub)

        message = socket.recv()
        self.drain_gaze(sub)

        logger.info(f"Received reply from OpenPose server: {message.decode()}")
        print(f"Received reply [ {message} ]")

    def turn(self, turn_order, object_order, turn_n, robot_turn_n, config_n, condition_n):

        # Robot turn: no gaze tracking
        if turn_order[turn_n] == 'y':
            #self.home_the_robot()
            block_shape = int(object_order[robot_turn_n])
            if robot_turn_n == 0:
                self.move_block(block_1_start, block_1_target, turn_n, block_shape)
            elif robot_turn_n == 1:
                self.move_block(block_2_start, block_2_target, turn_n, block_shape)
            elif robot_turn_n == 2:
                self.move_block(block_3_start, block_3_target, turn_n, block_shape)
            return 1

        # --- Human turns: P (right) / B (left) ---
        gaze_enabled = (
            (condition_n == 1 and (config_n == 1 or config_n == 2)) or
            (condition_n == 2 and (config_n == 3 or config_n == 4))
        )

        # ===== RIGHT PLAYER TURN =====
        if turn_order[turn_n] == 'p':

            if gaze_enabled:
                # Initial reaction: pick current mutual gazer and look
                self.drain_gaze(sub)
                self.last_gaze_update_time = time.time()
            else:
                self.home_the_robot()

            # --- Arm Extended phase ---
            print("Sending PR Arm Extended request...")
            message = self.wait_openpose_reply_with_gaze("PR: Arm Extended Check", gaze_enabled)

            #if gaze_enabled:
            #    self.home_the_robot()
            #    # or self.look_at_tower()
#
            # --- Arm Retracted phase ---
            print("Sending PR Arm Retracted request...")
            message = self.wait_openpose_reply_with_gaze("PR: Arm Retracted Check", gaze_enabled)

            #if gaze_enabled:
            #    self.home_the_robot()

            return 0

        # ===== LEFT PLAYER TURN =====
        elif turn_order[turn_n] == 'b':

            if gaze_enabled:
                # Initial reaction
                self.drain_gaze(sub)
                self.last_gaze_update_time = time.time()
            else:
                self.home_the_robot()

            # --- Arm Extended phase ---
            print("Sending PL Arm Extended request...")
            message = self.wait_openpose_reply_with_gaze("PL: Arm Extended Check", gaze_enabled)

            #if gaze_enabled:
            #    self.home_the_robot()
            #    # or self.look_at_tower()

            # --- Arm Retracted phase ---
            print("Sending PL Arm Retracted request...")
            message = self.wait_openpose_reply_with_gaze("PL: Arm Retracted Check", gaze_enabled)

            #if gaze_enabled:
            #    self.home_the_robot()

            return 0


    #def turn(self, turn_order, object_order, turn_n, robot_turn_n, config_n, condition_n):
#
    #    if turn_order[turn_n] == 'y':
    #        self.home_the_robot()
    #        block_shape = int(object_order[robot_turn_n])
    #        if robot_turn_n == 0:
    #            self.move_block(block_1_start, block_1_target, turn_n, block_shape)
    #        elif robot_turn_n == 1:
    #            self.move_block(block_2_start, block_2_target, turn_n, block_shape)
    #        elif robot_turn_n == 2:
    #            self.move_block(block_3_start, block_3_target, turn_n, block_shape)
    #        #self.home_the_robot()
    #        return 1
#
    #    elif turn_order[turn_n] == 'p':
#
    #        if (condition_n == 1 and (config_n == 1 or config_n == 2)) or (condition_n == 2 and (config_n == 3 or config_n == 4)):
    #            self.drain_gaze(sub)
    #            self.look_at_mutual_gazer()
    #            #self.look_at_person_1()
    #        else:
    #            #self.look_at_ground() DISCOMENT
    #            self.home_the_robot()
    #        # Wait until person 1 starts doing his action
    #        print("Sending PR Arm Extended request...")
    #        self.drain_gaze(sub)
    #        socket.send_string("PR: Arm Extended Check")
    #        logger.info("Sent 'PR: Arm Extended Check' to OpenPose server.")
    #        self.drain_gaze(sub)
#
    #        message = socket.recv()
    #        self.drain_gaze(sub)
    #        print(f"Received reply [ {message} ]")
    #        logger.info(f"Received reply from OpenPose: {message.decode()}")
#
    #        if (condition_n == 1 and (config_n == 1 or config_n == 2)) or (condition_n == 2 and (config_n == 3 or config_n == 4)):
    #            self.home_the_robot()
#
    #            #self.look_at_tower() DISCOMENT
#
    #        # Wait until person 1 finishes his action
    #        print("Sending PR Arm Retracted request...")
#
    #        self.drain_gaze(sub)
    #        socket.send_string("PR: Arm Retracted Check")
    #        logger.info("Sent 'PR: Arm Retracted Check' to OpenPose server.")
    #        self.drain_gaze(sub)
#
    #        message = socket.recv()
    #        self.drain_gaze(sub)
    #        print(f"Received reply [ {message} ]")
    #        logger.info(f"Received reply from OpenPose: {message.decode()}")
#
#
    #        if (condition_n == 1 and (config_n == 1 or config_n == 2)) or (condition_n == 2 and (config_n == 3 or config_n == 4)):
    #            self.home_the_robot()
#
    #        return 0
#
    #    elif turn_order[turn_n] == 'b':
#
    #        if (condition_n == 1 and (config_n == 1 or config_n == 2)) or (condition_n == 2 and (config_n == 3 or config_n == 4)):
#
    #            self.drain_gaze(sub)
    #            self.look_at_mutual_gazer()
    #            #self.look_at_person_2()
    #        else:
    #            #self.look_at_ground() DISCOMENT
    #            self.home_the_robot()
#
    #        # Wait until person 2 starts doing his action
    #        print("Sending PL Arm Extended request...")
#
    #        self.drain_gaze(sub)
    #        socket.send_string("PL: Arm Extended Check")
    #        logger.info("Sent 'PL: Arm Extended Check' to OpenPose server.")
    #        self.drain_gaze(sub)
#
    #        message = socket.recv()
    #        self.drain_gaze(sub)
    #        print(f"Received reply [ {message} ]")
    #        logger.info(f"Received reply from OpenPose: {message.decode()}")
#
#
    #        if (condition_n == 1 and (config_n == 1 or config_n == 2)) or (condition_n == 2 and (config_n == 3 or config_n == 4)):
    #            self.home_the_robot()
#
    #            #self.look_at_tower() DISCOMENT
#
    #        # Wait until person 2 finishes his action
    #        print("Sending PL Arm Retracted request...")
#
#
    #        self.drain_gaze(sub)
    #        socket.send_string("PL: Arm Retracted Check")
    #        logger.info("Sent 'PL: Arm Retracted Check' to OpenPose server.")
    #        self.drain_gaze(sub)
#
    #        message = socket.recv()
    #        self.drain_gaze(sub)
    #        print(f"Received reply [ {message} ]")
    #        logger.info(f"Received reply from OpenPose: {message.decode()}")
#
#
#
    #        if (condition_n == 1 and (config_n == 1 or config_n == 2)) or (condition_n == 2 and (config_n == 3 or config_n == 4)):
    #            self.home_the_robot()
#
    #        return 0
        
    def _recv_gaze_packet(self, sub, flags=0):
        """
        Receive one gaze message from the SUB socket and return the payload bytes.

        Handles both:
          - multipart: [topic, payload]
          - single-part: [payload]

        Returns:
          payload (bytes) or None if no message is available.
        """
        try:
            parts = sub.recv_multipart(flags=flags)
        except zmq.Again:
            return None

        if not parts:
            return None
        if len(parts) == 1:
            # Just JSON payload
            return parts[0]
        else:
            # First frame is topic, last is payload
            return parts[-1]

        
    def calibrate_mutual_gaze(self, sub, duration_s=10.0, min_samples=30):
        """
        Automatically estimate yaw/pitch bias between camera and robot.
        Assumes that during the calibration period the participant is
        looking at the ROBOT'S HEAD (not the camera). The mean yaw/pitch
        observed in camera frame during this period is used as the bias:
            yaw_robot   = yaw_cam   - yaw_bias_deg
            pitch_robot = pitch_cam - pitch_bias_deg
        Call this once before starting the configurations.
        """
        import time
        start = time.time()
        yaw_vals = []
        pitch_vals = []
        print("\n=== MUTUAL GAZE CALIBRATION ===")
        print("Please look at the ROBOT'S HEAD and stay still for "
              f"{duration_s:.1f} seconds...")
        logger.info("Starting mutual gaze calibration for %.1f seconds", duration_s)
        while (time.time() - start) < duration_s and not rospy.is_shutdown():
            payload = self._recv_gaze_packet(sub, flags=zmq.NOBLOCK)
            if payload is None:
                time.sleep(0.02)
                continue

            try:
                data = json.loads(payload)
                people = data.get("people", [])
                for e in people:
                    yaw = e.get("yaw_deg")
                    pitch = e.get("pitch_deg")
                    if yaw is None or pitch is None:
                        continue
                    yaw_vals.append(float(yaw))
                    pitch_vals.append(float(pitch))
            except Exception as ex:
                logger.warning("Calibration: failed to parse gaze payload: %s", ex)

            time.sleep(0.02)
        if len(yaw_vals) < min_samples:
            logger.warning(
                "Calibration collected only %d samples (< %d). Bias may be noisy.",
                len(yaw_vals), min_samples
            )
        if yaw_vals:
            self.yaw_bias_deg = sum(yaw_vals) / len(yaw_vals)
            self.pitch_bias_deg = sum(pitch_vals) / len(pitch_vals)
            logger.info(
                "Calibration done: yaw_bias=%.3f deg, pitch_bias=%.3f deg",
                self.yaw_bias_deg, self.pitch_bias_deg
            )
            print(f"Calibration done.")
            print(f"  yaw_bias   = {self.yaw_bias_deg:.2f} deg")
            print(f"  pitch_bias = {self.pitch_bias_deg:.2f} deg\n")
        else:
            logger.error("Calibration failed: no valid gaze samples collected.")
            print("Calibration failed: no gaze samples received.\n")



    def drain_gaze(self, sub, max_msgs=5, log=True):
        """
        Non-blocking: drain up to max_msgs gaze packets from SUB socket.
        Optionally logs the latest payload, but does NOT store any gaze state.
        """
        drained = 0
        last = None
        while drained < max_msgs:
            payload = self._recv_gaze_packet(sub, flags=zmq.NOBLOCK)
            if payload is None:
                break
            last = payload
            drained += 1

        if not log or last is None:
            return

        try:
            data = json.loads(last)
            frame = data.get("frame_idx")
            ts = data.get("timestamp")
            people = data.get("people", [])
            logger.info("Gaze rx: frame_idx=%s ts=%s people=%d", frame, ts, len(people))
            for e in people:
                v = e.get("dir_cam")
                yaw = e.get("yaw_deg")
                pitch = e.get("pitch_deg")
                pid = e.get("id")
                lbl = e.get("label")
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    logger.info(
                        "Gaze3D rx frame=%s id=%s label=%s v=[%.3f, %.3f, %.3f] yaw=%.1f pitch=%.1f",
                        frame, pid, lbl, v[0], v[1], v[2], yaw, pitch
                    )
                else:
                    logger.info(
                        "Gaze3D rx frame=%s id=%s label=%s v=<missing> yaw=%.1f pitch=%.1f",
                        frame, pid, lbl, yaw, pitch
                    )
        except Exception as ex:
            logger.warning("Failed to parse gaze payload (%d bytes): %s", len(last), ex)



    

    
    def main(self):
        # For testing purposes
      
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python")
            logger.info("Deleted ROS parameter /kortex_examples_test_results/cartesian_poses_with_notifications_python (if it existed).")
        except:
            logger.info("ROS parameter /kortex_examples_test_results/cartesian_poses_with_notifications_python did not exist or could not be deleted.")
            pass

        if success:

            logger.info("Starting main robot control loop.")

            #*******************************************************************************
            # Make sure to clear the robot's faults else it won't move if it's already in fault
            success &= self.robot_clear_faults()
            #*******************************************************************************

            #*******************************************************************************
            # Set the reference frame to "Mixed"
            success &= self.robot_set_cartesian_reference_frame()

            #*******************************************************************************
            # Subscribe to ActionNotification's from the robot to know when a cartesian pose is finished
            success &= self.subscribe_to_a_robot_notification()

            #*******************************************************************************

            self.home_the_robot()

            condition_n = int(sys.argv[1])

            argc = len(sys.argv)
            config_n = 1

            while config_n <= argc:

                sys.stdout.write(" Press Enter to start configuration " + str(config_n) + "\n")
                sys.stdout.flush()
                input()
                logger.info(f"User pressed Enter. Continuing with configuration {config_n}")
                print(" Continuing with configuration", config_n)



                # Get all the objects starting and target positions
                info = self.parse_information(config_n + 1)
                turn_order = info[0]
                object_order = info[1]
                turn_n = 0
                robot_turn_n = 0

                print(" About to send turn_order:", turn_order)
                logger.info("Sent turn_order and waiting for response from OpenPose server.")
                self.send_turn_order(turn_order)
                print("Sent turn_order and waiting for response...")

                if config_n == 1:
                    # (optional) wait for first gaze packet if you added that helper
                    # self.wait_for_first_gaze(sub, timeout_s=10.0)
            
                    self.calibrate_mutual_gaze(sub, duration_s=10.0, min_samples=30)
            


                while turn_n < len(turn_order)-1:
                    logger.info(f"Executing turn {turn_n} out of {len(turn_order)-1}.")
                    if self.turn(turn_order, object_order, turn_n, robot_turn_n, config_n, condition_n) == 1:
                        robot_turn_n += 1
                        logger.info(f"Robot completed its turn. Robot turn count: {robot_turn_n}")
                    turn_n += 1

                logger.info(f"Configuration number {config_n} finished.")
                print("Configuration number " + str(config_n) +" finished")


                self.drain_gaze(sub)
                socket.send_string("Configuration finished")
                logger.info("Sent 'Configuration finished' to OpenPose server.")
                self.drain_gaze(sub)

                message = socket.recv()
                self.drain_gaze(sub)
                logger.info(f"Received reply from OpenPose server: {message.decode()}")

                config_n += 1
            #Connected to OpenPose server
            # Movement finished, send to home position
            success &= self.zero_the_robot()
            

            success &= self.all_notifs_succeeded

            success &= self.all_notifs_succeeded

        
        # For testing purposes
        rospy.set_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python", success)
        logger.info(f"Setting ROS parameter /kortex_examples_test_results/cartesian_poses_with_notifications_python to {success}.")

        if not success:
            rospy.logerr("The example encountered an error.")
            logger.error("The build_tower example encountered an error during execution.")

if __name__ == "__main__":
    ex = BuildTower()
    ex.main()
