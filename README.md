# Yolov5 Minimal Node for ROS-Noetic
It's a simplified version of https://github.com/mats-robotics/yolov5_ros

It just publishes Object type by String type, so you can use string type in your control source code. 

## Prerequisites
Ubuntu 20.04, ROS-Noetic, Py-torch
and I use Python 3.8

## How to Install
1. Create your ros workspace
    ```
    mkdir -p <your_ws>/src
    
    cd <your_ws>
    ```
    
2. Install this in <your_ws>/src by
    ```
    git clone https://github.com/McCatony/yolov5_ros
    ```

    Then yolov5 folder and detection.py will be in <your_ws>/src/yolov5_ros/src
3. Change file mode by
    ```
    chmod 777 detection.py
    ```
    
4. Install Yolo_v5 in <your_ws>/src/yolov5_ros/src by https://github.com/ultralytics/yolov5.

   And do requirements in https://github.com/ultralytics/yolov5.
5. Execute $ catkin_make in your_ws, not anywhere
6. Run by
    ```
    rosrun yolov5_ros detection.py
    ```

## How to launch
0. Source ROS setup.bash
1. Run 'roscore' in your first terminal
2. Run or Launch any camera node in your second terminal
3. Find your topic type. For example,
    ```
    rostopic info /camera/color/image_raw
    ```
    
4. Revise detection.py for your image message type in line 74
5. Run detection.py
6. In your third terminal, you can find the class name which is detected by
    ```
    rostopic echo /object
    ```
    
   And if you don't like node and topic name, then you can edit node and topic name in line 73 and 75
7. If you want to use your yolov5 weights, edit line 43 and pose your weights at your_ws/src/yolov5_ros. Also, you can change confidence threshold at line 36
8. (Optional) If you want to see what is captured by a camera, disable the comment in line 78, 115~117. But, there will be no bounding boxes. 
