
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def callback(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            rospy.signal_shutdown("ESC pressed")
    except Exception as e:
        rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node("camera_viewer", anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber("/camera/color/image_raw", Image, callback)
    rospy.loginfo("Viewing /camera/color/image_raw")
    rospy.spin()
    cv2.destroyAllWindows()
