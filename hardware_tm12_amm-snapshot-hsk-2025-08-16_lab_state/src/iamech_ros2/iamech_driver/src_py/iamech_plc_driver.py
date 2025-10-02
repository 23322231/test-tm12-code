#!/usr/bin/env python3

import pyads, time, math

class iAMech_AMR():
    MAX_WHEEL_SPEED_MM_S = 200  # 輪子速度上限 (mm/s)
    MM_PER_METER = 1000.0       # 單位轉換係數

    def __init__(self,
                 SENDER_AMS = '1.2.3.4.1.1',
                 HOST_IP = '192.168.100.30',
                 PLC_IP = '192.168.100.100',
                 PLC_AMS_ID = '192.168.100.100.1.1',
                 PORT = 801,
                 PLC_USERNAME = 'Administrator',
                 PLC_PASSWORD = '1',
                 ROUTE_NAME = 'RouteToMyPC'):

        # 網路參數
        self.SENDER_AMS = SENDER_AMS
        self.HOST_IP = HOST_IP
        self.PLC_IP = PLC_IP
        self.PLC_AMS_ID = PLC_AMS_ID
        self.PORT = PORT
        self.PLC_USERNAME = PLC_USERNAME
        self.PLC_PASSWORD = PLC_PASSWORD
        self.ROUTE_NAME = ROUTE_NAME

        # 機器人參數
        self._wheelbase_mm = 588.0

        # 初始化狀態變數
        self._plc = None
        self.last_time = None
        self.x = self.y = self.theta = 0.0
        self.last_right = self.last_left = None

        # 初始化 PLC 參數列表和資料類型
        self._init_plc_parameters()

    def connect(self):
        try:
            pyads.open_port()
            pyads.add_route_to_plc(self.SENDER_AMS,
                                self.HOST_IP,
                                self.PLC_IP,
                                self.PLC_USERNAME,
                                self.PLC_PASSWORD,
                                route_name=self.ROUTE_NAME)
            pyads.close_port()

            # 創建連線
            self._plc = pyads.Connection(self.PLC_AMS_ID, self.PORT, self.PLC_IP)
            self._plc.open()
            self.last_time = time.time()
            print("成功連接到 PLC")

            return True
        except Exception as e:
            print(f"連接 PLC 失敗: {e}")
            return False

    def disconnect(self):
        """關閉與 PLC 的連線"""
        try:
            # 停止馬達
            self.write_velocity(0.0, 0.0)
            self.servo_off()
            print("已停止馬達")
        except Exception as e:
            print(f"停止馬達失敗: {e}")

        # 關閉連線
        if hasattr(self, '_plc') and self._plc:
            try:
                self._plc.close()
                print("已關閉 PLC 連線")
            except Exception as e:
                print(f"關閉 PLC 連線失敗: {e}")

    def read(self, names):
        """
        從 PLC 讀取單個或多個變數

        Args:
            names: 變數名稱或名稱列表

        Returns:
            單個值或值的列表
        """
        if isinstance(names, (list, tuple)): # Input type is list or tuple.
            return [self.__read_by_name(name) for name in names]
        return self.__read_by_name(names)

    def write(self, names, values):
        """
        向 PLC 寫入單個或多個變數

        Args:
            names: 變數名稱或名稱列表
            values: 寫入值或值列表
        """
        if isinstance(names, (list, tuple)):
            for name, value in zip(names, values):
                self.__write_by_name(name, value)
        else:
            self.__write_by_name(names, values)

    def servo_on(self):
        self.write(".bSLAM_ServeON", 1)

    def servo_off(self):
        self.write(".bSLAM_ServeON", 0)

    def write_velocity(self, linear_x, angular_z):
        # 轉成毫米每秒，再寫入 PLC
        vr_mm_s = int((2*linear_x*self.MM_PER_METER + angular_z*self._wheelbase_mm) / 2)
        vl_mm_s = int((2*linear_x*self.MM_PER_METER - angular_z*self._wheelbase_mm) / 2)
        vr_mm_s = max(min(vr_mm_s, self.MAX_WHEEL_SPEED_MM_S), -self.MAX_WHEEL_SPEED_MM_S)
        vl_mm_s = max(min(vl_mm_s, self.MAX_WHEEL_SPEED_MM_S), -self.MAX_WHEEL_SPEED_MM_S)

        self.write([".SLAM_R[2]", ".SLAM_L[2]"], [vr_mm_s, vl_mm_s])

    def _init_plc_parameters(self):
        """初始化 PLC 參數列表和資料類型"""
        # 參數列表
        self.parameters_list = [".bSLAM_ServeON"] + \
                            [f".SLAM_R[{i}]" for i in range(11, 19)] + \
                            [f".SLAM_L[{i}]" for i in range(11, 19)]

        # 資料類型字典
        self.PLC_DATA_TYPE = {
            ".bSLAM_ServeON": pyads.PLCTYPE_BOOL,
            ".SLAM_R[2]": pyads.PLCTYPE_DINT,
            ".SLAM_L[2]": pyads.PLCTYPE_DINT,
        }

        # 批量添加參數類型
        for i in range(11, 19):
            self.PLC_DATA_TYPE[f".SLAM_R[{i}]"] = pyads.PLCTYPE_DINT
            self.PLC_DATA_TYPE[f".SLAM_L[{i}]"] = pyads.PLCTYPE_DINT

    def __del__(self):
        try:
            self.disconnect()  # 重用現有方法
        except:
            pass  # 確保析構函數不會引發異常

    def __read_by_name(self, name):
        """從 PLC 讀取單個變數"""
        try:
            if name not in self.PLC_DATA_TYPE:
                raise ValueError(f"未知的 PLC 變數類型: {name}")
            return self._plc.read_by_name(name, self.PLC_DATA_TYPE[name])
        except Exception as e:
            print(f"PLC 讀取錯誤. 變數: {name}, 錯誤: {e}")
            return None

    def __write_by_name(self, name, value):
        """向 PLC 寫入單個變數"""
        try:
            if name not in self.PLC_DATA_TYPE:
                raise ValueError(f"未知的 PLC 變數類型: {name}")
            self._plc.write_by_name(name, value, self.PLC_DATA_TYPE[name])
        except Exception as e:
            print(f"PLC 寫入錯誤. 變數: {name}, 錯誤: {e}")

