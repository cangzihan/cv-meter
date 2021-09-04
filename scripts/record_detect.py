# python3 record_detect.py

import sys
import cv2
import time
import rospy
import numpy as np
import os
import tensorflow as tf
from PIL import Image

from yolo import YOLO

read_path = "/home/zihan/ros/records/record4"
save_result = True
save_path = "/home/zihan/ros/pics/"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo = YOLO(use_sort=True)
cv_img = 0
fps = 0.0
save_id = 0
# Define the image size scaling factor
scaling_factor = 0.75

def main():
    save_id = 0
    fps = 0

    color_list = []
    depth_list = []
    file_list = os.listdir(read_path)
    file_list.sort()
    for file_name in file_list:
        if 'jpg' in file_name:
            color_list.append(file_name)
        else:
            depth_list.append(file_name)

    for i in range(len(color_list)):
        # Read the current frame
        frame = cv2.imread(read_path+'/'+color_list [i])
        frame_depth = cv2.imread(read_path+'/'+depth_list [i], cv2.IMREAD_UNCHANGED)

        t1 = time.time()
        #frame = color_list [i]
        #frame_depth = depth_list [i]

        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测当前帧
        frame, results = yolo.detect_image(frame, depth_pic=frame_depth)
        frame = np.array(frame)

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        pos_person = []
        pos_manipulator = []
        for result in results:
            if result[3] == "person":
                pos_person.append(np.array(result[:3]))
            elif result[3] == "manipulator":
                pos_manipulator.append(np.array(result[:3]))

        num_distance = 0
        if len(pos_person) * len(pos_manipulator) > 0:
            for i in range(len(pos_person)):
                for j in range(len(pos_manipulator)):
                    if pos_person[i][2] == -1 or pos_manipulator[j][2] == -1:
                         continue
                    distance = np.sqrt(np.sum(np.square(pos_person[i]-pos_manipulator[j])))
                    if distance < 1.0:
                        print("Too close! Distance:", distance)
                        frame = cv2.putText(frame, "Too close! Distance: %.2f"%(distance), (20, 80+num_distance*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        print("Distance: ", distance)
                        frame = cv2.putText(frame, "Distance:  %.2f"%(distance), (20,80+num_distance*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    num_distance+=1

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("record",frame)

        if save_result:
            path = save_path + "%04d_color.jpg" % save_id
            save_id += 1
            if save_id == 2000:
                save_id = 0
            cv2.imwrite(path, frame)
        cv2.waitKey(1)


if __name__ == '__main__':
     main()
