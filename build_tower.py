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
import threading
import copy
import ctypes
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

if not gaze_logger.hasHandlers():
    gaze_file_handler = logging.FileHandler(gaze_log_filename)
    gaze_formatter = logging.Formatter('%(asctime)s - %(message)s')
    gaze_file_handler.setFormatter(gaze_formatter)
    gaze_logger.addHandler(gaze_file_handler)


# ==============================
# Linux per-thread CPU affinity
# ==============================

def pin_to_core(core_id: int):
    """
    Pin the *current thread* to a given CPU core on Linux using sched_setaffinity.
    If anything fails, we just log a warning and continue normally.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # cpu_set_t as a single unsigned long bitmask (works for low core counts)
        mask = ctypes.c_ulong(1 << core_id)
        res = libc.sched_setaffinity(
            0,  # pid = 0 => "calling thread"
            ctypes.sizeof(mask),
            ctypes.byref(mask)
        )
        if res != 0:
            err = ctypes.get_errno()
            logger.warning(f"pin_to_core({core_id}) failed: errno={err}")
        else:
            logger.info(f"Pinned current thread to core {core_id}")
    except Exception as e:
        logger.warning(f"pin_to_core({core_id}) failed with exception: {e}")


# ==============================
# OpenPoseThread: ALL ZMQ HERE
# ==============================
import queue

class OpenPoseThread(threading.Thread):
    """
    Dedicated thread that owns ALL ZeroMQ sockets and handles:
      - REQ/REP on tcp://localhost:5555   (commands: turn order, PR/PL checks, etc.)
      - SUB     on tcp://localhost:5556   (continuous gaze stream)

    Robot thread API:
      - send_request(cmd_str, timeout=None) -> reply bytes or None
      - get_latest_gaze() -> deep copy of last gaze dict or None

    This thread can be pinned to a specific core using core_id.
    """
    def __init__(self, req_endpoint="tcp://localhost:5555",
                 sub_endpoint="tcp://localhost:5556",
                 core_id=None):
        super(OpenPoseThread, self).__init__(daemon=True)

        self.req_endpoint = req_endpoint
        self.sub_endpoint = sub_endpoint
        self.core_id = core_id

        self.context = zmq.Context()
        self.req_socket = None
        self.sub_socket = None
        self.poller = None

        self.cmd_queue = queue.Queue()
        self.reply_queue = queue.Queue()

        self._latest_gaze = None
        self._gaze_lock = threading.Lock()

        self.stop_event = threading.Event()
        

    def connect_sockets(self):
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(self.req_endpoint)
        logger.info("OpenPoseThread: Connected REQ to %s", self.req_endpoint)
        print(f"[OpenPoseThread] Connected REQ to {self.req_endpoint}")

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.RCVHWM, 10)
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.connect(self.sub_endpoint)
        logger.info("OpenPoseThread: Connected SUB to %s", self.sub_endpoint)
        print(f"[OpenPoseThread] Connected SUB to {self.sub_endpoint}")

        self.poller = zmq.Poller()
        self.poller.register(self.sub_socket, zmq.POLLIN)
        self.poller.register(self.req_socket, zmq.POLLIN)

    def _recv_gaze_packet(self, flags=0):
        try:
            parts = self.sub_socket.recv_multipart(flags=flags)
        except zmq.Again:
            return None

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        else:
            return parts[-1]

    def _update_latest_gaze_from_payload(self, payload):
        try:
            data = json.loads(payload)
        except Exception as ex:
            logger.warning("OpenPoseThread: failed to parse gaze payload: %s", ex)
            return

        with self._gaze_lock:
            self._latest_gaze = data

        # Optional logging
        frame = data.get("frame_idx")
        ts = data.get("timestamp")
        people = data.get("people", [])
        logger.info("OpenPoseThread: gaze frame_idx=%s ts=%s people=%d", frame, ts, len(people))
        for e in people:
            v = e.get("dir_cam")
            yaw = e.get("yaw_deg")
            pitch = e.get("pitch_deg")
            pid = e.get("id")
            lbl = e.get("label")
            if isinstance(v, (list, tuple)) and len(v) == 3:
                logger.info(
                    "OpenPoseThread: Gaze3D frame=%s id=%s label=%s v=[%.3f, %.3f, %.3f] yaw=%.1f pitch=%.1f",
                    frame, pid, lbl, v[0], v[1], v[2], yaw, pitch
                )
            else:
                logger.info(
                    "OpenPoseThread: Gaze3D frame=%s id=%s label=%s v=<missing> yaw=%.1f pitch=%.1f",
                    frame, pid, lbl, yaw, pitch
                )

    def _poll_gaze_only(self, timeout_ms=10):
        if self.poller is None:
            return
        socks = dict(self.poller.poll(timeout=timeout_ms))
        if self.sub_socket in socks and socks[self.sub_socket] & zmq.POLLIN:
            payload = self._recv_gaze_packet(flags=zmq.NOBLOCK)
            if payload is not None:
                gaze_logger.info("Gaze rx (idle): %s", payload.decode("utf-8", errors="ignore"))
                self._update_latest_gaze_from_payload(payload)

    def _wait_for_reply_with_gaze(self):
        """
        After sending a REQ, wait until a reply arrives.
        While waiting, keep reading SUB for gaze and updating _latest_gaze.
        """
        while not self.stop_event.is_set():
            socks = dict(self.poller.poll(timeout=50))  # 50 ms

            # Prioritize REQ reply
            if self.req_socket in socks and socks[self.req_socket] & zmq.POLLIN:
                msg = self.req_socket.recv()
                logger.info("OpenPoseThread: received reply: %s", msg.decode())
                return msg

            # Gaze updates
            if self.sub_socket in socks and socks[self.sub_socket] & zmq.POLLIN:
                payload = self._recv_gaze_packet(flags=zmq.NOBLOCK)
                if payload is not None:
                    gaze_logger.info("Gaze rx (during REQ): %s", payload.decode("utf-8", errors="ignore"))
                    self._update_latest_gaze_from_payload(payload)

        return None

    def run(self):
        # Pin this thread to a given core (if requested)
        if self.core_id is not None:
            pin_to_core(self.core_id)

        try:
            self.connect_sockets()
        except Exception as e:
            logger.critical("OpenPoseThread: failed to connect sockets: %s", e)
            return

        while not self.stop_event.is_set():
            # If we have a command, process it; otherwise just keep gaze fresh
            try:
                cmd_str = self.cmd_queue.get(timeout=0.1)
            except queue.Empty:
                self._poll_gaze_only(timeout_ms=10)
                continue

            if cmd_str is None:
                break  # shutdown

            logger.info("OpenPoseThread: sending cmd: %s", cmd_str)
            self.req_socket.send_string(cmd_str)

            reply = self._wait_for_reply_with_gaze()
            if reply is not None:
                self.reply_queue.put(reply)

        # Cleanup
        try:
            self.req_socket.close()
            self.sub_socket.close()
            self.context.term()
        except Exception:
            pass

    # ===== Public API for robot thread =====

    def send_request(self, cmd_str, timeout=None):
        """
        Synchronous: enqueue cmd, wait for reply.
        """
        self.cmd_queue.put(cmd_str)
        try:
            reply = self.reply_queue.get(timeout=timeout)
        except queue.Empty:
            logger.error("OpenPoseThread: timeout waiting for reply for cmd=%s", cmd_str)
            return None
        return reply

    def get_latest_gaze(self):
        """
        Return a deep copy of latest gaze dict or None.
        """
        with self._gaze_lock:
            if self._latest_gaze is None:
                return None
            return copy.deepcopy(self._latest_gaze)

    def stop(self):
        self.stop_event.set()
        self.cmd_queue.put(None)  # wake thread


# ==============================
# Kortex + game logic
# ==============================

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

            self.waiting_for_action = False


            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 7)
            self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

            rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " +
                          str(self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " +
                          str(self.is_gripper_present))
            logger.info(f"Robot parameters: robot_name={self.robot_name}, "
                        f"dof={self.degrees_of_freedom}, gripper_present={self.is_gripper_present}")

            # Action notifications
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic",
                                                     ActionNotification,
                                                     self.cb_action_topic)
            self.last_action_notif_type = None

            # Services
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
            self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name,
                                                                    SetCartesianReferenceFrame)
            logger.info(f"Service initialized: {set_cartesian_reference_frame_full_name}")

            activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(
                activate_publishing_of_action_notification_full_name,
                OnNotificationActionTopic
            )
            logger.info(f"Service initialized: {activate_publishing_of_action_notification_full_name}")

            send_gripper_command_full_name = '/' + self.robot_name + '/base/send_gripper_command'
            rospy.wait_for_service(send_gripper_command_full_name)
            self.send_gripper_service = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)
            logger.info(f"Service initialized: {send_gripper_command_full_name}")

            # Gaze bias + throttling
            self.yaw_bias_deg = 0.0
            self.pitch_bias_deg = 0.0
            self.last_gaze_update_time = 0.0   # for optional 3s gating

            # OpenPose thread pinned to core 1
            self.openpose = OpenPoseThread(
                req_endpoint="tcp://localhost:5555",
                sub_endpoint="tcp://localhost:5556",
                core_id=1  # <-- OpenPose ZMQ thread on core 1
            )
            self.openpose.start()
            logger.info("OpenPoseThread started (core 1).")

        except Exception as e:
            self.is_init_success = False
            logger.error(f"Initialization failed: {e}", exc_info=True)
        else:
            self.is_init_success = True
            logger.info("BuildTower class initialized successfully.")

    # =======================
    # Kortex helpers
    # =======================

    def cb_action_topic(self, notif):
        # Only latch notifications when we are explicitly waiting
        if self.waiting_for_action:
            self.last_action_notif_type = notif.action_event


    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if self.last_action_notif_type == ActionEvent.ACTION_END:
                rospy.loginfo("Received ACTION_END notification")
                logger.info("Robot action ended.")
                return True
            elif self.last_action_notif_type == ActionEvent.ACTION_ABORT:
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
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        self.last_action_notif_type = None
        try:
            res = self.read_action(req)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call ReadAction")
            logger.error(f"Failed to call ReadAction for home: {e}")
            return False
        else:
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
        req = ReadActionRequest()
        req.input.identifier = self.ZERO_ACTION_IDENTIFIER
        self.last_action_notif_type = None
        try:
            res = self.read_action(req)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call ReadAction")
            logger.error(f"Failed to call ReadAction for zero: {e}")
            return False
        else:
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
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED

        try:
            self.set_cartesian_reference_frame(req)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call ExecuteAction")
            logger.error(f"Failed to call ExecuteAction: {e}")
            return False
        else:
            rospy.loginfo("Set the cartesian reference frame successfully")
            logger.info("Cartesian reference frame set successfully.")
            rospy.sleep(0.25)
            return True

    def subscribe_to_a_robot_notification(self):
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
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo("Sending the gripper command...")
        logger.info(f"Sending gripper command: value={value}")

        try:
            self.send_gripper_service(req)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call SendGripperCommand")
            logger.error(f"Failed to call SendGripperCommand: {e}")
            return False
        else:
            time.sleep(0.5)
            logger.info(f"Gripper command {value} sent successfully.")
            return True

    # =======================
    # Game file + motions
    # =======================

    def parse_information(self, argv_n):
        logger.info(f"Parsing information from file: {sys.argv[argv_n]}")
        try:
            f = open(sys.argv[argv_n], 'r')
        except FileNotFoundError:
            logger.error(f"File not found: {sys.argv[argv_n]}")
            print(f"File not found: {sys.argv[argv_n]}")
            sys.exit(1)

        # y = robot; p = right; b = left
        turn_order = f.readline()
        # 1 = square; 2 = rectangle; 3 = semicircle; 4 = bridge
        object_order = f.readline()

        logger.info(f"Parsed turn_order: {turn_order}")
        logger.info(f"Parsed object_order: {object_order}")

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
        my_cartesian_speed.translation = translation
        my_cartesian_speed.orientation = orientation
        cons_pose.constraint.oneof_type.speed.append(my_cartesian_speed)
        logger.debug(f"Set speed for pose: translation={translation}, orientation={orientation}")

    def go_to_position(self, pos, wait=True):
        logger.info(
            f"Attempting to go to position: x={pos.target_pose.x}, y={pos.target_pose.y}, "
            f"z={pos.target_pose.z}, theta_x={pos.target_pose.theta_x}, "
            f"theta_y={pos.target_pose.theta_y}, theta_z={pos.target_pose.theta_z}"
        )
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

        if not wait:
            # Fire-and-forget (good for head/look poses)
            logger.info("Pose sent (non-blocking).")
            return True

        rospy.loginfo("Waiting for pose to finish...")
        logger.info("Pose action initiated, waiting for completion.")
        self.waiting_for_action = True
        ok = self.wait_for_action_end_or_abort()
        self.waiting_for_action = False
        return ok
    
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
        logger.info(f"Moving block from start {pos1.target_pose.z} to "
                    f"target {pos2.target_pose.z}. Turn: {turn_n}, Block Shape: {block_shape}")

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

        pos1.target_pose.z += 0.045 + turn_n * 0.045
        self.go_to_position(pos1)

        pos2.target_pose.z += 0.045
        self.go_to_position(pos2)

        pos2.target_pose.z -= 0.045
        self.go_to_position(pos2)

        self.control_gripper(0)

        pos2.target_pose.z += 0.045
        self.go_to_position(pos2)

        return True

    # =======================
    # Head poses
    # =======================

    def look_at_tower(self):
        tower_pos = ConstrainedPose()
        tower_pos.target_pose.x = 0.50
        tower_pos.target_pose.y = 0
        tower_pos.target_pose.z = 0.30
        tower_pos.target_pose.theta_x = 110
        tower_pos.target_pose.theta_y = 0
        tower_pos.target_pose.theta_z = 90
        logger.info("Robot moving to look at tower.")
        self.go_to_position(tower_pos, wait=False)

    def look_at_person_1(self):
        person_1_pos = ConstrainedPose()
        person_1_pos.target_pose.x = 0.5
        person_1_pos.target_pose.y = 0.08
        person_1_pos.target_pose.z = 0.50
        person_1_pos.target_pose.theta_x = 80
        person_1_pos.target_pose.theta_y = 0
        person_1_pos.target_pose.theta_z = 105
        logger.info("Robot moving to look at person on the right.")
        self.go_to_position(person_1_pos, wait=False)

    def look_at_person_2(self):
        person_2_pos = ConstrainedPose()
        person_2_pos.target_pose.x = 0.5
        person_2_pos.target_pose.y = -0.08
        person_2_pos.target_pose.z = 0.50
        person_2_pos.target_pose.theta_x = 80
        person_2_pos.target_pose.theta_y = 0
        person_2_pos.target_pose.theta_z = 75
        logger.info("Robot moving to look at person on the left.")
        self.go_to_position(person_2_pos, wait=False)

    def look_at_ground(self):
        ground_pose = ConstrainedPose()
        ground_pose.target_pose.x = 0.3
        ground_pose.target_pose.y = 0
        ground_pose.target_pose.z = 0.1
        ground_pose.target_pose.theta_x = 180
        ground_pose.target_pose.theta_y = 0
        ground_pose.target_pose.theta_z = 90
        logger.info("Robot moving to look at ground.")
        self.go_to_position(ground_pose, wait=False)

    # =======================
    # Gaze decision (stateless)
    # =======================

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
            raw_yaw = e.get("yaw_deg")
            raw_pitch = e.get("pitch_deg")
            if pid not in ("left", "right"):
                continue
            if raw_yaw is None or raw_pitch is None:
                continue

            yaw = float(raw_yaw) - self.yaw_bias_deg
            pitch = float(raw_pitch) - self.pitch_bias_deg

            if abs(yaw) <= yaw_thr and abs(pitch) <= pitch_thr:
                if abs(yaw) < best_yaw_abs:
                    best_yaw_abs = abs(yaw)
                    best_id = pid

        return best_id

    def look_once_from_latest_gaze(self):
        """
        Pull ONE latest gaze snapshot from OpenPoseThread and move head accordingly.
        Stateless: only uses the current snapshot.
        """
        data = self.openpose.get_latest_gaze()
        if data is None:
            logger.info("look_once_from_latest_gaze: no gaze available, looking at tower.")
            self.look_at_tower()
            return

        target = self._decide_target_from_data(data)
        logger.info("look_once_from_latest_gaze: target=%s", target)

        if target == "right":
            self.look_at_person_1()
        elif target == "left":
            self.look_at_person_2()
        else:
            self.look_at_tower()

    def maybe_update_head_from_gaze(self, gaze_enabled, min_interval_s=3.0):
        """
        Optional gating: only update head from gaze if at least min_interval_s passed.
        Uses a single, current snapshot (no memory / averaging).
        """
        if not gaze_enabled:
            return
        now = time.time()
        if now - self.last_gaze_update_time >= min_interval_s:
            self.look_once_from_latest_gaze()
            self.last_gaze_update_time = now

    # =======================
    # Calibration using OpenPoseThread
    # =======================

    def calibrate_mutual_gaze(self, duration_s=10.0, min_samples=30):
        """
        Uses OpenPoseThread.latest_gaze to estimate yaw/pitch bias.
        """
        start = time.time()
        yaw_vals = []
        pitch_vals = []
        print("\n=== MUTUAL GAZE CALIBRATION ===")
        print("Please look at the ROBOT'S HEAD and stay still for "
              f"{duration_s:.1f} seconds...")
        logger.info("Starting mutual gaze calibration for %.1f seconds", duration_s)

        while (time.time() - start) < duration_s and not rospy.is_shutdown():
            data = self.openpose.get_latest_gaze()
            if data is None:
                time.sleep(0.02)
                continue

            people = data.get("people", [])
            for e in people:
                yaw = e.get("yaw_deg")
                pitch = e.get("pitch_deg")
                if yaw is None or pitch is None:
                    continue
                yaw_vals.append(float(yaw))
                pitch_vals.append(float(pitch))
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

    # =======================
    # OpenPose REQ helpers
    # =======================

    def wait_openpose_reply_with_gaze(self, request_str, gaze_enabled):
        """
        Send a request to the OpenPose server via OpenPoseThread and wait for reply.
        The OpenPoseThread itself keeps updating gaze while waiting.

        After receiving the reply, we may optionally update the head from the
        latest gaze snapshot (stateless).
        """
        logger.info("Sending to OpenPose (via thread): %s", request_str)
        print(f"Sending request (via OpenPoseThread): {request_str}")
        msg = self.openpose.send_request(request_str, timeout=10.0)
        if msg is None:
            print("Warning: no reply from OpenPose for", request_str)
            return None
        print(f"Received reply [ {msg} ]")

        # After a successful reply, optionally update head (snapshot, stateless)
        self.maybe_update_head_from_gaze(gaze_enabled, min_interval_s=3.0)
        return msg

    def send_turn_order(self, turn_order):
        logger.info(f"Sending turn order to OpenPose server: {turn_order}")
        print("Sending turn order...")
        msg = self.openpose.send_request(turn_order, timeout=10.0)
        if msg is None:
            print("Warning: no reply to turn order!")
            logger.error("No reply from OpenPose to turn order.")
            return
        logger.info(f"Received reply from OpenPose server: {msg.decode()}")
        print(f"Received reply [ {msg} ]")

    # =======================
    # Turn logic
    # =======================

    def turn(self, turn_order, object_order, turn_n, robot_turn_n, config_n, condition_n):

        # Robot turn: no gaze tracking
        if turn_order[turn_n] == 'y':
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
                # Reset timer so first head update can happen immediately if desired
                self.last_gaze_update_time = 0.0
                self.maybe_update_head_from_gaze(gaze_enabled, min_interval_s=0.0)
            else:
                self.home_the_robot()

            # --- Arm Extended phase ---
            print("Sending PR Arm Extended request...")
            _ = self.wait_openpose_reply_with_gaze("PR: Arm Extended Check", gaze_enabled)

            # --- Arm Retracted phase ---
            print("Sending PR Arm Retracted request...")
            _ = self.wait_openpose_reply_with_gaze("PR: Arm Retracted Check", gaze_enabled)

            return 0

        # ===== LEFT PLAYER TURN =====
        elif turn_order[turn_n] == 'b':

            if gaze_enabled:
                self.last_gaze_update_time = 0.0
                self.maybe_update_head_from_gaze(gaze_enabled, min_interval_s=0.0)
            else:
                self.home_the_robot()

            # --- Arm Extended phase ---
            print("Sending PL Arm Extended request...")
            _ = self.wait_openpose_reply_with_gaze("PL: Arm Extended Check", gaze_enabled)

            # --- Arm Retracted phase ---
            print("Sending PL Arm Retracted request...")
            _ = self.wait_openpose_reply_with_gaze("PL: Arm Retracted Check", gaze_enabled)

            return 0

    # =======================
    # Main
    # =======================

    def main(self):
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python")
            logger.info("Deleted ROS parameter /kortex_examples_test_results/cartesian_poses_with_notifications_python (if it existed).")
        except Exception:
            logger.info("ROS parameter /kortex_examples_test_results/cartesian_poses_with_notifications_python did not exist or could not be deleted.")
            pass

        if success:
            logger.info("Starting main robot control loop.")

            success &= self.robot_clear_faults()
            success &= self.robot_set_cartesian_reference_frame()
            success &= self.subscribe_to_a_robot_notification()

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
                    self.calibrate_mutual_gaze(duration_s=10.0, min_samples=30)

                while turn_n < len(turn_order) - 1:
                    logger.info(f"Executing turn {turn_n} out of {len(turn_order) - 1}.")
                    if self.turn(turn_order, object_order, turn_n, robot_turn_n,
                                 config_n, condition_n) == 1:
                        robot_turn_n += 1
                        logger.info(f"Robot completed its turn. Robot turn count: {robot_turn_n}")
                    turn_n += 1

                logger.info(f"Configuration number {config_n} finished.")
                print("Configuration number " + str(config_n) + " finished")

                # Tell OpenPose config finished
                msg = self.openpose.send_request("Configuration finished", timeout=10.0)
                if msg is None:
                    logger.error("No reply to 'Configuration finished'")
                else:
                    logger.info(f"Received reply from OpenPose server: {msg.decode()}")

                config_n += 1

            success &= self.zero_the_robot()
            success &= self.all_notifs_succeeded

        rospy.set_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python", success)
        logger.info(f"Setting ROS parameter /kortex_examples_test_results/cartesian_poses_with_notifications_python to {success}.")

        if not success:
            rospy.logerr("The example encountered an error.")
            logger.error("The build_tower example encountered an error during execution.")


if __name__ == "__main__":
    # Pin the *main* robot/control thread to core 0
    pin_to_core(0)
    logger.info("Main BuildTower thread pinned to core 0.")

    ex = BuildTower()
    try:
        ex.main()
    finally:
        # Ensure OpenPose thread stops cleanly
        if hasattr(ex, "openpose") and ex.openpose is not None:
            ex.openpose.stop()
            ex.openpose.join(timeout=2.0)
