#!/usr/bin/env python3

import sys
import serial
import binascii
import time

import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from grpr2f85_ifaces.srv import GetGripperStatus, \
                                SetGripperState \
                                #Reset, \


class Gripper():
    def __init__(self, usbPort=1):
        self.usbPort = usbPort
        if (self.setup(self.usbPort) & self.set_gripper_state()):
            pass
        else:
            raise ConnectionError(f'no gripper found on usbPort{self.usbPort}')

    def __mycrc(self, input):
        crc = 0xffff
        for byte in input:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xa001
                else:
                    crc = crc >> 1
        crc = (((crc << 8) | (crc >> 8)) &
               0xFFFF).to_bytes(2, byteorder='big')
        return crc

    def setup(self,usbPort):
        try:
            self.ser = serial.Serial(port=f'/dev/ttyUSB{usbPort}', baudrate=115200,
                                     timeout=1, parity=serial.PARITY_NONE,
                                     stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
            self.ser.write(b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30')
            data_raw = self.ser.readline()
            data = binascii.hexlify(data_raw, ' ')
            return (data != b'')
        except Exception as e:
            print(e)
            return False

    def close(self):
        if hasattr(self, 'ser') and self.ser:
            try:
                self.ser.close()
                print("串口已關閉")
                return True
            except Exception as e:
                print(f"關閉串口時出錯: {e}")
                return False
        return True

    def reset(self):
        try:
            self.ser.write(b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30')
            data_raw = self.ser.readline()
            data = binascii.hexlify(data_raw, ' ')
            self.ser.write(b'\x09\x10\x03\xE8\x00\x03\x06\x01\x00\x00\x00\x00\x00\x72\xE1')
            timeout = time.time() + 5.0
            while time.time() < timeout:
                self.ser.write(b'\x09\x03\x07\xD0\x00\x01\x85\xCF')
                data_raw = self.ser.readline()
                if data_raw == b'\x09\x03\x02\x11\x00\x55\xD5':
                    print('Activate Not Complete')
                    pass
                elif data_raw == b'\x09\x03\x02\x31\x00\x4C\x15':
                    print('Activate Complete')
                    return True
            return False
        except Exception as e:
            print(e)
            return False

    def get_gripper_status(self):
        self.ser.write(b'\x09\x03\x07\xD0\x00\x03\x04\x0E')
        data_raw = self.ser.readline()  # bytes too slow
        data_show = binascii.hexlify(data_raw, ' ')
        print(f'Response: {data_show}')
        gripper_status_mask = b'\xFF'
        gripper_status = bytes([data_raw[3] & gripper_status_mask[0]])
        gripper_status = int.from_bytes(gripper_status, 'big')
        if gripper_status & 0b00001000 == 0b00001000 and\
                gripper_status & 0b11000000 == 0b00000000:
            msg = f'No Object Detect (gripper moving)'
            print(msg)
            status_code=0
            status_description=msg
        elif gripper_status & 0b11000000 == 0b01000000:
            msg = f'Object Detect (opening)'
            print(msg)
            status_code=1
            status_description=msg
        elif gripper_status & 0b11000000 == 0b10000000:
            msg = f'Object Detect (closing)'
            print(msg)
            status_code=2
            status_description=msg
        elif gripper_status & 0b11000000 == 0b11000000 or\
            (gripper_status & 0b00001000 == 0b00000000 and
             gripper_status & 0b11000000 == 0b00000000):
            msg = f'No Object Detect (gripper stop)'
            print(msg)
            status_code=3
            status_description=msg

        return True, status_code, status_description

    def set_gripper_state(self, position=0, speed=255, force=255, wait_time=0):
        position_byte = min(255,max(0,int(position+0.5))).to_bytes(1,'big')  # full open
        speed_byte = min(255,max(0,int(speed+0.5))).to_bytes(1,'big')  # 00:min;FF:max
        force_byte = min(255,max(0,int(force+0.5))).to_bytes(1,'big')  # 00:min;FF:max
        time.sleep(max(int(wait_time+0.5),0))  # calibrate:\x15
        command = b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00' +\
                position_byte + speed_byte + force_byte
        command+=self.__mycrc(command)
        print('set state:', command)
        self.ser.write(command)
        data_raw = self.ser.readline()
        data = binascii.hexlify(data_raw, ' ') # bytes
        print('Response:', data)
        return True

class Gripper_ROS2_Node(Node, Gripper):
    def __init__(self, usbPort=1, name= 'grpr2f85_driver', parent=None):
        Gripper.__init__(self, usbPort)
        Node.__init__(self, name)
        self.srv_reset = self.create_service(Trigger, '~/reset', self.reset_callback)
        self.srv_set_gripper_state = self.create_service(SetGripperState,
                                                         '~/set_gripper_state',
                                                          self.set_gripper_state_callback)
        self.srv_get_gripper_status = self.create_service(GetGripperStatus,
                                                         '~/get_gripper_status',
                                                          self.get_gripper_status_callback)
        self.get_logger().info(f'Initial ok')

    def destroy_node(self):
        self.get_logger().info('關閉夾爪節點並釋放串口資源')
        self.close()  # 調用關閉串口的方法
        super().destroy_node()

    def reset_callback(self,request, response):
        if self.reset():
            response.success = True
        else:
            response.success = False
        return response

    def get_gripper_status_callback(self, request, response):
        response.ok, response.status_code, response.result = self.get_gripper_status()
        return response

    def set_gripper_state_callback(self, request, response):
        response.ok = self.set_gripper_state(request.position,
                                             request.speed,
                                             request.force,
                                             request.wait_time)
        _, response.status_code , response.result = self.get_gripper_status()
        return response

def get_port_from_argv():
    valid_argv = [i for i in sys.argv if 'usb_port:=' in i]
    if len(valid_argv)>=1:
        return valid_argv[0][10:]
    else:
        print('Didn\'t find argument: usb_port:=xxx')
        return ''

def main(args=None):
    rclpy.init(args=args)
    try:
        port = get_port_from_argv()
        node = Gripper_ROS2_Node(usbPort=port)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exception:
        raise exception
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
