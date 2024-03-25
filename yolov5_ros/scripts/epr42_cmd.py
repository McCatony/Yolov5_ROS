import rospy as rp
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class Erp42_Cmd : 
    def __init__(self) :
        rp.init_node("erp42_cmd_node")
        rp.Subscriber('/object', String, self.yolo_callback)
        self.erp_pub = rp.Publisher('cmd_vel', Twist, queue_size=10)
        self.cmd_vel_msg = Twist()
        self.cmd_vel_msg.linear.x
        self.cmd_vel_msg.angular.z
    
    def yolo_callback(self, msg) : 
        # msg is String and infromation about environment near the erp-42
        # below command line is well only if stop line is nearby
        if msg == "Red" : 
            # below code is for if left turn is needed. so, for example
            # if msg in "something direct turn left" :
                if msg in "left" :
                    self.cmd_vel_msg.linear.x = 1
                    self.cmd_vel_msg.angular.z = 1
                else : 
                    self.cmd_vel_msg.linear.x = 0
            # below code is for if right turn is needed. So, for example
            # if msg in "something direct turn right" : 
                # below code is for if obstacle is detected by lidar.
                if msg in "obstacle" : 
                    self.cmd_vel_msg.linear.x = 0
                    self.cmd_vel_msg.angular.z = 0
                else :
                    self.cmd_vel_msg.linear.x = 1
                    self.cmd_vel_msg.angular.z = -1
        elif msg == "Yellow" : 
            # slow down
            pass
            # or
            # if stop line is nearby
            #   self.cmd_vel_msg.linear.x = 0
            # else 
            #   stop
        else : 
            # if msg == "Green"
            self.cmd_vel_msg.linear.x = 1
        
        self.erp_pub.publish(self.cmd_vel_msg)

def main() : 
    erp42_cmd = Erp42_Cmd()
    rp.spin()

if __name__ == "__main__" : 
    main()