# python3 record_detect.py

import jsonfiler
import sys
import cv2
import time
import rospy
import numpy as np
import os
import tensorflow as tf
from PIL import Image

from yolo import YOLO

read_path = "/home/zihan/ros/records/data"
save_result = True
save_path = "/home/zihan/ros/datas/"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

save_id = 0
# Define the image size scaling factor
scaling_factor = 0.75

location_list = [chr(i) for i in range(ord("A"),ord("Y")+1)]

real_data = jsonfiler.load(read_path +"/real_data.json")
err = []
def main():
    save_id = 0
    fps = 0

    color_list = []
    depth_list = []
    for location in location_list:
        file_list = os.listdir(read_path+'/'+location)
        file_list.sort()
        for file_name in file_list:
            if 'jpg' in file_name:
                color_list.append(read_path+'/'+location+'/'+file_name)
            elif 'depth.png' in file_name:
                depth_list.append(read_path+'/'+location+'/'+file_name)
    for i in range(len(color_list)):
        print(color_list[i], depth_list[i])

    yolo = YOLO(use_sort=False)

    save_dict = {}
    for h in range(len(color_list)):
        # Read the current frame
        frame = cv2.imread(color_list [h])
        frame_depth = cv2.imread(depth_list [h], cv2.IMREAD_UNCHANGED)

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

                    if color_list [h][-16:] not in save_dict.keys():
                        save_dict[color_list [h][-16:]] = {
                        'value': round(distance,6),
                         'err': distance - real_data[color_list[h][-16]]
                         }
                        err.append(distance - real_data[color_list[h][-16]])

                    if distance < 1.0:
                        print("Too close! Distance:", distance)
                        frame = cv2.putText(frame, "Too close! Distance: %.2f"%(distance), (20, 80+num_distance*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        print("Distance: ", distance)
                        frame = cv2.putText(frame, "Distance:  %.2f"%(distance), (20,80+num_distance*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    num_distance+=1

        cv2.imshow("record",frame)

        if save_result:
            path = save_path + "%04d_color.jpg" % save_id
            save_id += 1
            cv2.imwrite(path, frame)
        cv2.waitKey(1)

    print("MSA:", np.sum(np.array(err)**2)/len(err))
    jsonfiler.dump(save_dict, path+"a.json", indent=4)


if __name__ == '__main__':
     main()
