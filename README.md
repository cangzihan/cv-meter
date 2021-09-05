# CV-Meter
[ROS Project] Distance measurement between human and robot by Kinect V2.
I modified the YOLO v4 code of this repository: https://github.com/bubbliiiing/yolov4-tf2

## Requirements
- A GPU with at least 8G of RAM（RTX3060, RTX3090, etc.）
- Ubuntu 18.04
- ROS Melodic
- Install CV Bridge in your ROS workspace: https://github.com/code-iai/iai_kinect2
- Python3
- Python Lib: opencv, numpy, tensorflow-gpu>2.2.0

My operating environment: RTX3090, cuda:11.4.1, tensorflow:2.6.0, cudnn:8.2.2.26, GPU driver version:470.63.01

## Usage
My YOLO v4 model data file will be avaliable later.

Under ～/catkin_ws/src【Your ROS workspace path】：
```
git clone https://github.com/cangzihan/cv-meter.git
cd ..
catkin_make
```

### Open 3 linux terminals:
#### In the first Linux terminal, start the roscore:
```
roscore
```

#### In the second Linux terminal, start the cv bridge:
```
roslaunch kinect2_bridge kinect2_bridge.launch
```

#### In the third Linux terminal:
1. Solve the cv_bridge problem in Ros melodic python3 environment. 

I refer to this blog：https://blog.csdn.net/weixin_42675603/article/details/107785376
```
cd ~/catkin_ws/src/cv-meter/script/      # Your ROS workspace
source install/setup.bash --extend
```

2.run the YOLO v4 node：
```
rosrun cv-meter Kinect2_yolo.py
```
