#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Char
import sys
import termios
import tty
import select
import threading
#要遙控時, 請輸入
#   ros2 run tm12_amm keyboard_pub.py --ros-args -r /keyboard_manual:=robot/keyboard_manual

class KeyboardPublisher(Node):
    """鍵盤輸入發布器 - 用於 TM12 AMM 手動控制"""

    def __init__(self):
        super().__init__('keyboard_publisher')

        # 創建發布器
        self.publisher = self.create_publisher(Char, 'keyboard_manual', 10)

        # 檢查是否在終端環境中
        self.is_tty = sys.stdin.isatty()
        self.old_settings = None

        if self.is_tty:
            try:
                # 保存原始終端設置
                self.old_settings = termios.tcgetattr(sys.stdin)

                # 設置終端為非阻塞模式
                tty.setraw(sys.stdin.fileno())

                self.get_logger().info('鍵盤發布器已啟動（終端模式）\n')
            except Exception as e:
                self.get_logger().error(f'設置終端模式失敗: {e}')
                self.is_tty = False
        else:
            self.get_logger().warn('非終端環境，鍵盤輸入可能不可用')

        # 控制變量
        self.running = True

        # 啟動鍵盤監聽線程
        self.keyboard_thread = threading.Thread(target=self._keyboard_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        self._print_help()

    def _keyboard_listener(self):
        """鍵盤監聽器"""
        if not self.is_tty:
            self.get_logger().warn('非終端環境，鍵盤監聽不可用')
            return

        while self.running:
            try:
                # 減少超時時間，讓退出更快響應
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1)

                    # 處理 ESC 鍵
                    if ord(key) == 27:
                        # 檢查是否是 ESC 序列
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            key += sys.stdin.read(2)
                            if key == '\x1b[':
                                continue
                        else:
                            # 單獨的 ESC 鍵
                            self._publish_key(27)
                            continue

                    # 處理 Ctrl+C
                    if ord(key) == 3:  # Ctrl+C
                        self.get_logger().info('收到 Ctrl+C，正在退出...')
                        self.running = False
                        # 發送中斷信號給主線程
                        import signal
                        import os
                        os.kill(os.getpid(), signal.SIGINT)

                    # 發布按鍵
                    self._publish_key(ord(key))

            except Exception as e:
                self.get_logger().error(f'鍵盤監聽錯誤: {e}')
                self.running = False
                break

    def _publish_key(self, key_code: int):
        """發布按鍵"""
        msg = Char()
        msg.data = key_code
        self.publisher.publish(msg)

        # 轉換為字符顯示
        if key_code < 128:
            char = chr(key_code)
            if char.isprintable():
                self.get_logger().info(f'發布按鍵: {char} (ASCII: {key_code})')
            else:
                self.get_logger().info(f'發布按鍵: ASCII {key_code}')
        else:
            self.get_logger().info(f'發布按鍵: {key_code}')

    def _print_help(self):
        """打印幫助信息"""
        mode_text = '(終端模式)' if self.is_tty else '(非終端模式)'

        help_text = (
            f"{'=' * 50}\n"
            f"鍵盤手動控制說明 {mode_text}\n"
            f"{'=' * 50}\n\n"
            "機械手臂控制:\n"
            "    q/a     - +X/-X (對角線)\n"
            "    w/s     - +Y/-Y (對角線)\n"
            "    e/d     - +Z/-Z\n"
            "    r/f     - +Roll/-Roll\n"
            "    t/g     - +Pitch/-Pitch\n"
            "    y/h     - +Yaw/-Yaw\n\n"
            "夾爪控制:\n"
            "    z       - 夾爪關閉\n"
            "    x       - 夾爪打開\n\n"
            "速度控制:\n"
            "    k       - 增加速度\n"
            "    l       - 減少速度\n\n"
            "其他功能:\n"
            "    u       - 拍照\n"
            "    j       - 回到原位\n"
            "    i       - 顯示幫助\n"
            "    ESC     - 退出手動模式\n"
            "    Ctrl+C  - 退出程序\n\n"
            "注意: 請確保 TM12 AMM 節點已啟動並啟用手動模式"
        )

        if not self.is_tty:
            help_text += "\n提示: 如需使用鍵盤輸入，請在終端中直接運行此節點"

        help_text += f"\n{'=' * 50}"

        # 使用 print 而不是 logger，避免時間戳和節點名稱干擾排版
        print(help_text)

        # 記錄到日誌中（簡化版本）
        self.get_logger().info(f"已顯示鍵盤控制說明 {mode_text}")

    def destroy_node(self):
        """清理資源"""
        self.running = False

        # 只有在終端模式下才嘗試恢復設置
        if self.is_tty and self.old_settings is not None:
            try:
                # 使用 TCSANOW 立即生效，而不是 TCSADRAIN
                termios.tcsetattr(sys.stdin, termios.TCSANOW, self.old_settings)
                self.get_logger().info('終端設置已恢復')
            except Exception as e:
                self.get_logger().error(f'恢復終端設置失敗: {e}')

        super().destroy_node()

def main(args=None):
    """主函數"""
    rclpy.init(args=args)

    keyboard_publisher = KeyboardPublisher()

    try:
        rclpy.spin(keyboard_publisher)
    except KeyboardInterrupt:
        pass  # 不需要額外的日誌，因為鍵盤監聽器已經輸出了
    finally:
        # 快速清理
        keyboard_publisher.running = False
        keyboard_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
