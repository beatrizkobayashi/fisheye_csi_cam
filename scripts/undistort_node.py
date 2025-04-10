#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class FisheyeUndistort:
    def __init__(self):
        self.bridge = CvBridge()

        # Load calibration parameters
        cam_name = rospy.get_param("~cam_name", "csi_cam_0") # Get camera name as a private parameter
        namespace = "/" + cam_name + "/"
        self.K = np.array(rospy.get_param(namespace + "camera_matrix/data")).reshape(3,3)
        self.D = np.array(rospy.get_param(namespace + "distortion_coefficients/data"))

        # Get image width and height from parameters
        self.image_width = rospy.get_param(namespace + "image_width")
        self.image_height = rospy.get_param(namespace + "image_height")

        # Create Undistortion Maps (pre-compute once)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.K,
            (self.image_width, self.image_height),
            cv2.CV_16SC2
        )

        # Setup subscribers/publishers
        self.sub = rospy.Subscriber(namespace + "image_raw", Image, self.image_cb)
        self.pub = rospy.Publisher("image_undistorted", Image, queue_size=10)

        rospy.loginfo("Fisheye undistortion node started.")

    def image_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            undistorted = cv2.remap(cv_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            self.pub.publish(self.bridge.cv2_to_imgmsg(undistorted, "bgr8"))
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    rospy.init_node('fisheye_undistort_node')
    undistort_node = FisheyeUndistort()
    rospy.spin()