# CV-Meter
[ROS Project] Distance measurement between human and robot by Kinect V2.

I modified the YOLO v4 code of this repository: https://github.com/bubbliiiing/yolov4-tf2

## Requirements
- A GPU with at least 8G of RAM（RTX3060, RTX3090, etc.）
- Kinect V2
- Ubuntu 18.04
- ROS Melodic
- Install CV Bridge in your ROS workspace: https://github.com/code-iai/iai_kinect2
- Python3
- Python Lib: opencv, numpy, tensorflow-gpu>2.2.0

#### My operating environment: 
| GPU | cuda | cudnn | tensorflow-gpu | GPU driver version |
| :-----: | :-----: | :------: | :------: | :------: |
| RTX3090 | 11.4.1 | 8.2.2.26 | 2.6.0 | 470.63.01 |

## Usage
My YOLO v4 model data file will be avaliable later.

### Install
Under ~/catkin_ws/src【Your ROS workspace path】：
#### clone my ROS project:
```
git clone https://github.com/cangzihan/cv-meter.git
```
or you can clone by:
```
git clone git://github.com/cangzihan/cv-meter.git
```

#### unzip crucial files and compile your ROS workspace:
```
cd cv-meter/scripts
unzip py3_cvbridge.zip
chmod a+x _setup_util.py
chmod a+x Kinect2_yolo.py
cd ../../..
catkin_make
```

### Open 3 linux terminals:
Before starting, please make sure cv_bridge is ready.
as your ROS workspace(like ~/catkin_ws/src) must include the following folders:
- cv-meter
- [iai_kinect2-master](https://github.com/code-iai/iai_kinect2)
- [vision_opencv](https://github.com/ros-perception/vision_opencv.git)
#### In the first Linux terminal, start the roscore:
```
roscore
```

#### In the second Linux terminal, start the cv bridge:
```
roslaunch kinect2_bridge kinect2_bridge.launch
```

#### In the third Linux terminal:
- 1. Solve the cv_bridge problem in Ros melodic python3 environment. 

I refer to this blog：https://blog.csdn.net/weixin_42675603/article/details/107785376
```
cd ~/catkin_ws/src/cv-meter/scripts          # Your ROS workspace
source setup.bash --extend
```

- 2. run the YOLO v4 node：

download my model data file(avaliable later), move the last2.h5 to the path 【.../cv-meter/scripts/logs/】

if your ROS workspace is not 【~/catkin_ws】, change the path from scripts/yolo.py to your path.

run the third ROS node:
```
rosrun cv-meter Kinect2_yolo.py
```
