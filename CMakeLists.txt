cmake_minimum_required(VERSION 3.0.2)
project(fisheye_csi_cam)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  image_transport
  gscam
)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    sensor_msgs 
    cv_bridge 
    image_transport 
    gscam
)

catkin_install_python(
  PROGRAMS 
    scripts/undistort_node.py  # Your main Python node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY 
  launch/
  config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

