#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from iamech_plc_driver import iAMech_AMR
from iamech_ifaces.srv import WPLC
from iamech_ifaces.msg import PLCStatus


class iAMech_AMR_ROS2_Node(Node, iAMech_AMR):
    def __init__(self):
        Node.__init__(self, 'iAMech_AMR_driver')
        self.declare_parameter('SENDER_AMS', '1.2.3.4.1.1')
        self.declare_parameter('HOST_IP', '192.168.100.30')
        self.declare_parameter('PLC_IP', '192.168.100.100')
        self.declare_parameter('PLC_AMS_ID', '192.168.100.100.1.1')
        self.declare_parameter('PORT', 801)
        self.declare_parameter('PLC_USERNAME', 'Administrator')
        self.declare_parameter('PLC_PASSWORD', '1')
        self.declare_parameter('ROUTE_NAME', 'RouteToMyPC')
        self.declare_parameter('fps', 60)

        self.SENDER_AMS = self.get_parameter('SENDER_AMS').value
        self.HOST_IP = self.get_parameter('HOST_IP').value
        self.PLC_IP = self.get_parameter('PLC_IP').value
        self.PLC_AMS_ID = self.get_parameter('PLC_AMS_ID').value
        self.PORT = self.get_parameter('PORT').value
        self.PLC_USERNAME = self.get_parameter('PLC_USERNAME').value
        self.PLC_PASSWORD = self.get_parameter('PLC_PASSWORD').value
        self.ROUTE_NAME = self.get_parameter('ROUTE_NAME').value

        iAMech_AMR.__init__(self,
                            self.SENDER_AMS,
                            self.HOST_IP,
                            self.PLC_IP,
                            self.PLC_AMS_ID,
                            self.PORT,
                            self.PLC_USERNAME,
                            self.PLC_PASSWORD,
                            self.ROUTE_NAME)
        self.connect()
        self.get_logger().info(f'Connected to PLC: {self.PLC_IP}')

        self.fps = self.get_parameter('fps').value

        self.timer = self.create_timer(1/self.fps, self.timer_callback)

        self.plc_status_pub_ = self.create_publisher(PLCStatus, '~/plc_status', 60)
        self.cmd_vel_sub_ = self.create_subscription(Twist, '~/cmd_vel', self.cmd_vel_callback, 10)

        self.write_plc_srv = self.create_service(WPLC, '~/write_plc', self.write_srv_callback)
        self.servo_on_srv = self.create_service(Trigger, '~/servo_on', self.servo_on_srv_callback)
        self.servo_off_srv = self.create_service(Trigger, '~/servo_off', self.servo_off_srv_callback)

        self.get_logger().info(f'iAMech_AMR node started')

    def get_status(self):
        parameters = self.read(self.parameters_list)
        status = PLCStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.serve_on = parameters[0]
        status.right.bready = parameters[1]
        status.right.bmoving = parameters[2]
        status.right.berror = parameters[3]
        status.right.pos = parameters[4]
        status.right.velocity = parameters[5]
        status.right.errorcode = parameters[6]
        status.right.temperature = parameters[7]
        status.right.volt = parameters[8]

        status.left.bready = parameters[9]
        status.left.bmoving = parameters[10]
        status.left.berror = parameters[11]
        status.left.pos = parameters[12]
        status.left.velocity = parameters[13]
        status.left.errorcode = parameters[14]
        status.left.temperature = parameters[15]
        status.left.volt = parameters[16]
        return status

    def cmd_vel_callback(self, msg):
        self.get_logger().info(f'Get new cmd_vel: linear.x: ({msg.linear.x}, angular.z: {msg.angular.z})')
        self.write_velocity(msg.linear.x, msg.angular.z)

    def write_srv_callback(self, request, response):
        self.write(request.name, request.value)
        return response

    def servo_on_srv_callback(self, request, response):
        self.servo_on()
        self.get_logger().info("Servo ON")
        response.success = True
        response.message = "Servo ON"
        return response

    def servo_off_srv_callback(self, request, response):
        self.servo_off()
        self.get_logger().info("Servo OFF")
        response.success = True
        response.message = "Servo OFF"
        return response

    def timer_callback(self):
        msg = PLCStatus()
        msg = self.get_status()
        self.plc_status_pub_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = iAMech_AMR_ROS2_Node()
        node.get_logger().info('iAMech AMR node started')
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down')
    except Exception as exception:
        node.get_logger().error(f'Unexpected error: {str(exception)}')
        raise exception
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()


