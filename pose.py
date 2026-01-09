#!/usr/bin/env python3
import os
import sys
import time
import rospy
import logging
import ctypes
from datetime import datetime

from kortex_driver.srv import *
from kortex_driver.msg import *

# ----------------------------
# Logging
# ----------------------------
if "RUN_TIMESTAMP" not in os.environ:
    os.environ["RUN_TIMESTAMP"] = datetime.now().strftime("%Y%m%d_%H%M")

timestamp = os.environ["RUN_TIMESTAMP"]
log_dir = os.path.join(os.path.dirname(__file__), "logs", timestamp)
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(log_dir, f"kinova_pose_runner_{timestamp}.log")
logger = logging.getLogger("PoseRunner")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    fh = logging.FileHandler(log_filename)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# ----------------------------
# CPU affinity (optional)
# ----------------------------
def pin_to_core(core_id: int):
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        mask = ctypes.c_ulong(1 << core_id)
        res = libc.sched_setaffinity(0, ctypes.sizeof(mask), ctypes.byref(mask))
        if res != 0:
            err = ctypes.get_errno()
            logger.warning(f"pin_to_core({core_id}) failed: errno={err}")
        else:
            logger.info(f"Pinned current thread to core {core_id}")
    except Exception as e:
        logger.warning(f"pin_to_core({core_id}) failed with exception: {e}")


# ----------------------------
# Pose Runner
# ----------------------------
class PoseRunner:
    def __init__(self):
        rospy.init_node("kinova_pose_runner_python")

        self.HOME_ACTION_IDENTIFIER = 2

        self.robot_name = rospy.get_param("~robot_name", "my_gen3")
        self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 7)
        self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

        rospy.loginfo(f"Using robot_name {self.robot_name}, dof={self.degrees_of_freedom}, gripper={self.is_gripper_present}")
        logger.info(f"Robot parameters: robot_name={self.robot_name}, dof={self.degrees_of_freedom}, gripper={self.is_gripper_present}")

        # Notifications
        self.waiting_for_action = False
        self.last_action_notif_type = None
        self.action_topic_sub = rospy.Subscriber(
            f"/{self.robot_name}/action_topic",
            ActionNotification,
            self.cb_action_topic
        )

        # Services
        clear_faults_full_name = f"/{self.robot_name}/base/clear_faults"
        read_action_full_name = f"/{self.robot_name}/base/read_action"
        execute_action_full_name = f"/{self.robot_name}/base/execute_action"
        set_cartesian_reference_frame_full_name = f"/{self.robot_name}/control_config/set_cartesian_reference_frame"
        activate_notifs_full_name = f"/{self.robot_name}/base/activate_publishing_of_action_topic"

        rospy.wait_for_service(clear_faults_full_name)
        rospy.wait_for_service(read_action_full_name)
        rospy.wait_for_service(execute_action_full_name)
        rospy.wait_for_service(set_cartesian_reference_frame_full_name)
        rospy.wait_for_service(activate_notifs_full_name)

        self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)
        self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)
        self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)
        self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name, SetCartesianReferenceFrame)
        self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_notifs_full_name, OnNotificationActionTopic)

        logger.info("All Kortex services initialized.")

    def cb_action_topic(self, notif):
        if self.waiting_for_action:
            self.last_action_notif_type = notif.action_event

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if self.last_action_notif_type == ActionEvent.ACTION_END:
                logger.info("Received ACTION_END")
                return True
            if self.last_action_notif_type == ActionEvent.ACTION_ABORT:
                logger.warning("Received ACTION_ABORT")
                return False
            time.sleep(0.01)

    def robot_clear_faults(self):
        try:
            self.clear_faults()
            rospy.sleep(2.5)
            logger.info("Faults cleared.")
            return True
        except rospy.ServiceException as e:
            logger.error(f"ClearFaults failed: {e}")
            return False

    def robot_set_cartesian_reference_frame(self):
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        try:
            self.set_cartesian_reference_frame(req)
            rospy.sleep(0.25)
            logger.info("Cartesian reference frame set.")
            return True
        except rospy.ServiceException as e:
            logger.error(f"SetCartesianReferenceFrame failed: {e}")
            return False

    def subscribe_to_a_robot_notification(self):
        req = OnNotificationActionTopicRequest()
        try:
            self.activate_publishing_of_action_notification(req)
            rospy.sleep(1.0)
            logger.info("Action notifications activated.")
            return True
        except rospy.ServiceException as e:
            logger.error(f"OnNotificationActionTopic failed: {e}")
            return False

    def home_the_robot(self):
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        self.last_action_notif_type = None

        try:
            res = self.read_action(req)
        except rospy.ServiceException as e:
            logger.error(f"ReadAction(home) failed: {e}")
            return False

        req2 = ExecuteActionRequest()
        req2.input = res.output
        try:
            self.execute_action(req2)
        except rospy.ServiceException as e:
            logger.error(f"ExecuteAction(home) failed: {e}")
            return False

        time.sleep(0.6)
        return True

    def set_speed(self, cons_pose, translation, orientation):
        my_cartesian_speed = CartesianSpeed()
        my_cartesian_speed.translation = translation
        my_cartesian_speed.orientation = orientation
        cons_pose.constraint.oneof_type.speed.append(my_cartesian_speed)

    def go_to_position(self, pos, translation_speed=1.0, orientation_speed=100.0):
        self.set_speed(pos, translation_speed, orientation_speed)

        req = ExecuteActionRequest()
        req.input.oneof_action_parameters.reach_pose.append(pos)
        req.input.name = "pose"
        req.input.handle.action_type = ActionType.REACH_POSE

        self.last_action_notif_type = None
        try:
            self.execute_action(req)
        except rospy.ServiceException as e:
            logger.error(f"ExecuteAction(reach_pose) failed: {e}")
            return False

        self.waiting_for_action = True
        ok = self.wait_for_action_end_or_abort()
        self.waiting_for_action = False
        return ok

    # ----------------------------
    # Config parsing (same format)
    # ----------------------------
    def parse_config_file(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)

        b1_start = ConstrainedPose()
        b1_target = ConstrainedPose()
        b2_start = ConstrainedPose()
        b2_target = ConstrainedPose()
        b3_start = ConstrainedPose()
        b3_target = ConstrainedPose()

        def read_pose(fh, pose_obj, block_name):
            line = fh.readline().strip()
            if line != block_name:
                return False
            pose_obj.target_pose.x = float(fh.readline())
            pose_obj.target_pose.y = float(fh.readline())
            pose_obj.target_pose.z = float(fh.readline())
            pose_obj.target_pose.theta_x = float(fh.readline())
            pose_obj.target_pose.theta_y = float(fh.readline())
            pose_obj.target_pose.theta_z = float(fh.readline())
            return True

        with open(filepath, "r") as f:
            # first two lines exist in your files but we don't need them here
            _turn_order = f.readline()
            _object_order = f.readline()

            read_pose(f, b1_start, "B1 start")
            read_pose(f, b1_target, "B1 target")
            read_pose(f, b2_start, "B2 start")
            read_pose(f, b2_target, "B2 target")
            read_pose(f, b3_start, "B3 start")
            read_pose(f, b3_target, "B3 target")

        poses = [
            ("B1 start", b1_start),
            ("B1 target", b1_target),
            ("B2 start", b2_start),
            ("B2 target", b2_target),
            ("B3 start", b3_start),
            ("B3 target", b3_target),
        ]
        return poses

    def run(self, config_path: str, dwell_s: float = 10.0):
        logger.info(f"Running pose sequence from config: {config_path}")

        ok = True
        ok &= self.robot_clear_faults()
        ok &= self.robot_set_cartesian_reference_frame()
        ok &= self.subscribe_to_a_robot_notification()
        ok &= self.home_the_robot()

        if not ok:
            rospy.logerr("Initialization steps failed; aborting.")
            return

        poses = self.parse_config_file(config_path)

        for name, pose in poses:
            if rospy.is_shutdown():
                break

            rospy.loginfo(f"Going to pose: {name}")
            logger.info(
                f"Going to {name}: "
                f"x={pose.target_pose.x:.3f} y={pose.target_pose.y:.3f} z={pose.target_pose.z:.3f} "
                f"rx={pose.target_pose.theta_x:.1f} ry={pose.target_pose.theta_y:.1f} rz={pose.target_pose.theta_z:.1f}"
            )

            ok = self.go_to_position(pose)
            if not ok:
                rospy.logerr(f"Failed to reach pose {name}; stopping.")
                logger.error(f"Failed to reach pose {name}; stopping.")
                return

            rospy.loginfo(f"Reached {name}. Waiting {dwell_s:.1f}s...")
            logger.info(f"Reached {name}. Dwell {dwell_s:.1f}s.")
            time.sleep(dwell_s)

        rospy.loginfo("Done with all poses.")
        logger.info("Completed all poses.")


if __name__ == "__main__":
    pin_to_core(0)

    if len(sys.argv) < 2:
        print("Usage: go_through_config_poses.py /path/to/config.txt [dwell_seconds]")
        sys.exit(1)

    config_path = sys.argv[1]
    dwell_s = float(sys.argv[2]) if len(sys.argv) >= 3 else 10.0

    runner = PoseRunner()
    runner.run(config_path, dwell_s=dwell_s)
