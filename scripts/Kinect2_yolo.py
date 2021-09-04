#!/usr/bin/env python3
# coding=utf-8
# license removed for brevity
# roslaunch kinect2_bridge kinect2_bridge.launch
# cd catkin_workspace/
# source install/setup.bash --extend
# rosrun distance_measurement_master Kinect2_yolo.py
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image as Image_ros
from cv_bridge import CvBridge, CvBridgeError

import sys
import cv2
import time
import rospy
import numpy as np
import os
import tensorflow as tf
from PIL import Image

from yolo import YOLO

save_path = "/home/zihan/ros/pics/"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo = YOLO()
cv_img = 0
fps = 0.0
save_id = 0
# Define the image size scaling factor
scaling_factor = 0.75

def callback(data):
    global cv_depth
    global bridge
    global fps
    global save_id
    t1 = time.time()
    cv_img = bridge.imgmsg_to_cv2(data, "bgr8")

    # 取出当前帧
    frame = cv_img
    frame_depth = cv_depth

    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测当前帧（文件已经读取完毕）
    frame, results = yolo.detect_image(frame, ros_mode=True, depth_pic=frame_depth)
    frame = np.array(frame)

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    pos_person = []
    pos_manipulator = []
    for result in results:
        if result[3] == "person":
            pos_person.append(np.array(result[:3]))
        elif result[3] == "manipulator":
            pos_manipulator.append(np.array(result[:3]))

    if len(pos_person) * len(pos_manipulator) > 0:
        for i in range(len(pos_person)):
            for j in range(len(pos_manipulator)):
                distance = np.sqrt(np.sum(np.square(pos_person[i]-pos_manipulator[j])))
                if distance < 0.5:
                    print("Too close! Distance:", distance)
                    frame = cv2.putText(frame, "Too close! Distance: %.2f"%(distance), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print("Distance: ", distance)
                    frame = cv2.putText(frame, "Distance:  %.2f"%(distance), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)
    if True:
        if save_id>=1000:
            path = save_path + "%d_color.jpg" % save_id
        elif save_id>=100:
            path = save_path + "0%d_color.jpg" % save_id
        elif save_id>=10:
            path = save_path + "00%d_color.jpg" % save_id
        else:
            path = save_path + "000%d_color.jpg" % save_id
        save_id += 1
        if save_id == 1000:
            save_id = 0
        cv2.imwrite(path, frame)
    cv2.waitKey(1)

def callback_depth(data):
    global bridge
    global cv_depth
    cv_depth = bridge.imgmsg_to_cv2(data, "passthrough")

if __name__ == '__main__':
     try:
         rospy.init_node('img_process_node', anonymous=True)
         bridge = CvBridge()
         rospy.Subscriber('/kinect2/qhd/image_color_rect', Image_ros, callback)
         rospy.Subscriber('/kinect2/qhd/image_depth_rect', Image_ros,  callback_depth)
         rospy.spin()
     except rospy.ROSInterruptException:
         pass
