#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Twist, Vector3, TransformStamped

from iamech_ifaces.msg import PLCStatus

class iAMech_Odom_ROS2_Node(Node):
    def __init__(self):
        # 參數宣告
        super().__init__('iAMech_AMR_Odom')
        self.declare_parameter('wheelbase', 588.0)
        self.declare_parameter('send_tf', True)
        self.declare_parameter('use_time_based', True)  # 是否使用時間差計算

        # 基本參數
        self.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # AMM current position
        self.L_mm = self.get_parameter('wheelbase').value  # Wheelbase
        self.send_tf = self.get_parameter('send_tf').value  # 是否發送 TF
        self.use_time_based = self.get_parameter('use_time_based').value  # 使用時間差計算

        # TF 廣播器
        self.tf_broadcaster = TransformBroadcaster(self)

        # 初始化 Odometry 訊息
        self.odom = Odometry()
        self.odom.header.frame_id = 'odom'
        self.odom.child_frame_id = 'iamech_base_link'

        # 發布者設定
        self.odom_pub = self.create_publisher(Odometry, '~/odom', 5)

        # 狀態追踪
        self.last_status = None

        # 日誌輸出計算方式
        calc_method = "時間差計算" if self.use_time_based else "位置差計算"
        self.get_logger().info(f'使用 {calc_method} 里程計')

        # 訂閱 PLC 狀態
        self.create_subscription(PLCStatus, 'iamech_driver/plc_status', self._odom_callback, 1)

    def _odom_pub(self, now, twist):
        quat = Quaternion()
        quat.x = 0.0
        quat.y = 0.0
        quat.z = math.sin(self.position[5] / 2.0)
        quat.w = math.cos(self.position[5] / 2.0)

        if self.send_tf:
            # 創建 TransformStamped 訊息
            t = TransformStamped()
            t.header.stamp = now.to_msg()
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base_link'

            # 設置變換
            t.transform.translation.x = self.position[0]
            t.transform.translation.y = self.position[1]
            t.transform.translation.z = self.position[2]
            t.transform.rotation.x = quat.x
            t.transform.rotation.y = quat.y
            t.transform.rotation.z = quat.z
            t.transform.rotation.w = quat.w

            # 發送變換
            self.tf_broadcaster.sendTransform(t)

        # 發布 Odometry 訊息
        self.odom.header.stamp = now.to_msg()
        self.odom.pose.pose.position.x = self.position[0]
        self.odom.pose.pose.position.y = self.position[1]
        self.odom.pose.pose.position.z = 0.0
        self.odom.pose.pose.orientation = quat
        self.odom.twist.twist = twist

        self.odom_pub.publish(self.odom)

    def _calculate_velocities(self, req):
        """計算線速度和角速度"""
        vel_x = (req.left.velocity + req.right.velocity) * math.cos(self.position[5]) / 2000.0
        vel_y = (req.left.velocity + req.right.velocity) * math.sin(self.position[5]) / 2000.0
        vel_w = 1.0 / self.L_mm * (req.right.velocity - req.left.velocity)
        return vel_x, vel_y, vel_w

    def _create_twist_msg(self, vel_x, vel_y, vel_w):
        """創建 Twist 訊息"""
        twist = Twist()
        twist.linear.x = vel_x
        twist.linear.y = vel_y
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = vel_w
        return twist

    def _odom_callback(self, req):
        """統一的里程計計算回調函數"""
        now = self.get_clock().now()

        # 初始化，第一次收到消息
        if self.last_status is None:
            self.last_status = req.header.stamp if self.use_time_based else req
            return

        # 使用時間差計算（原 _odom_jessica 方法）
        if self.use_time_based:
            # 計算時間差
            curr_stamp = req.header.stamp
            delta_t = (curr_stamp.sec - self.last_status.sec) + (curr_stamp.nanosec - self.last_status.nanosec) / 1e9

            # 計算速度
            vel_x, vel_y, vel_w = self._calculate_velocities(req)

            # 更新位置（時間積分）
            self.position[0] += vel_x * delta_t
            self.position[1] += vel_y * delta_t
            self.position[5] += vel_w * delta_t
            self.position[5] %= (math.pi * 2)

            # 更新狀態
            self.last_status = req.header.stamp

        # 使用位置差計算（原 _callback 方法）
        else:
            # 計算位置差
            diff_left_mm = req.left.pos - self.last_status.left.pos
            diff_right_mm = req.right.pos - self.last_status.right.pos

            # 更新位置
            self._compute_pose(diff_right_mm, diff_left_mm)

            # 更新狀態
            self.last_status = req

        # 計算當前速度（兩種方法都需要）
        vel_x, vel_y, vel_w = self._calculate_velocities(req)

        # 創建並發布 twist 訊息
        twist = self._create_twist_msg(vel_x, vel_y, vel_w)
        self._odom_pub(now, twist)

    def _compute_pose(self, diff_right_mm, diff_left_mm):
        # 不移動
        if diff_left_mm == 0 and diff_right_mm == 0:
            return

        dx_mm = 0
        dy_mm = 0
        dw_rad = 0

        # 直線
        if diff_right_mm == diff_left_mm:
            self.position[0] += diff_right_mm * math.cos(self.position[5]) / 1000
            self.position[1] += diff_right_mm * math.sin(self.position[5]) / 1000
            return

        # 原地旋轉
        elif diff_right_mm + diff_left_mm == 0:
            self.position[5] = (self.position[5] + 2.0 * diff_right_mm / self.L_mm) % (math.pi * 2)
            return

        else:
            # 左轉
            if abs(diff_left_mm) < abs(diff_right_mm):
                theta = (diff_right_mm - diff_left_mm) / float(self.L_mm)
                temp = self.L_mm / 2.0
                if diff_left_mm != 0:
                    temp += diff_left_mm / theta
                dw_rad = theta
                dx_ = temp * math.sin(theta)
                dy_ = temp * (1 - math.cos(theta))
            # 右轉
            else:
                theta = (diff_left_mm - diff_right_mm) / float(self.L_mm)
                temp = self.L_mm / 2.0
                if diff_right_mm != 0:
                    temp += diff_right_mm / theta
                dw_rad = -theta
                dx_ = temp * math.sin(theta)
                dy_ = temp * (math.cos(theta) - 1)

            dx_mm = dx_ * math.cos(self.position[5]) - dy_ * math.sin(self.position[5])
            dy_mm = dx_ * math.sin(self.position[5]) + dy_ * math.cos(self.position[5])

            self.position[0] += dx_mm / 1000
            self.position[1] += dy_mm / 1000
            self.position[5] = (self.position[5] + dw_rad) % (math.pi * 2)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = iAMech_Odom_ROS2_Node()
        node.get_logger().info('iAMech odometry node started')
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