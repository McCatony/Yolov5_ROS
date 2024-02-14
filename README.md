# Yolov5 Minimal Node for ROS-Noetic
It's a revised version of https://github.com/mats-robotics/yolov5_ros

It just publishes Object type by String type, so you can use string type in your control source code. 

## Prerequisites
Ubuntu 20.04, ROS-Noetic, Py-torch
and I use Python 3.8

## How to Install
1. Create your ros workspace

    $ mkdir -p your_ws/src
   
    $ cd your_ws

3. Create a package for Yolo

    $ catkin_create_pkg yolov5_ros rospy roscpp std_msgs sensor_msgs message_generation

4. Exchange CMakeLists.txt from default one at your_ws/src to this one 
5. Install Yolov5 in your_ws/src/yolov5_ros/src
6. Install this code in your_ws/src/yolov5_ros/src. Then yolov5 folder and detection.py will be in your_ws/src/yolov5_ros/src
7. Execute $ catkin_make in your_ws, not anywhere
8. Run

## How to launch
0. Source ROS setup.bash
1. Run 'roscore' in your first terminal
2. Run or Launch any camera node in your second terminal
3. Find your topic type. For example, '$ rostopic info /camera/color/image_raw'
4. Revise detection.py for your image message type in line 74
5. Run detection.py
6. In your third terminal, you can find the class name which is detected by '% rostopic echo /object. And if you don't like node and topic name, then you can edit node and topic name in line 73 and 75
7. If you want to use your yolov5 weights, edit line 43 and pose your weights at your_ws/src/yolov5_ros. Also, you can change confidence threshold at line 36
8. (Optional) If you want to see what is captured by a camera, then disable the comment in line 78, 115~117.
