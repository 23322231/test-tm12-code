#ros2 action send_goal /robot/arm/ai_action tm12_amm_interfaces/action/Dotask "{task: 'AI_Action', scenario: '', repeat_times: 1}"

# 1. 引入aruco到這裡
# 2. 
#


def _execute_ai_action_v2(self, scenario: str, repeat_times: int = 1):
        """執行 AI 動作"""
        try:
            self.get_logger().info(f'開始執行 AI 動作: "{scenario}"，重複 {repeat_times} 次')
            counter = 0
            while counter < repeat_times:
                if self.tm12_feedback and self.tm12_feedback.robot_error:
                    self.get_logger().warn('檢測到手臂錯誤，AI動作中止')
                    raise Exception('Robot error triggered')
                task_id = uuid.uuid4().hex
                self.get_logger().info(f'執行 AI 動作第 {counter+1}/{repeat_times} 次，任務 ID: {task_id}')

                # 移動到拍照位置

                self.robot_controller.move_to_pose(self.pose_take_photo_static)
                self.get_logger().info(f'debug1')
                self.robot_controller.wait_for_arrival(self.pose_take_photo_static)
                time.sleep(1)

                # 獲取機器人當前位置
                gripper_pose = self.tm12_feedback.tool_pose
                self.get_logger().info(f'debug2')

                # 調用 AI 推理
                ai_response = self._call_ai_inference(task_id)
                self.get_logger().info(f'debug3')

                if ai_response is None or not ai_response.success:
                    self.get_logger().error(f'AI 推理失敗，任務 ID: {task_id}')
                    if task_id:
                        self._report_success_to_ai(task_id, 0)
                    continue
                
                # 計算物體在基座中的位置
                object_pose_in_base = RobotPose.get_object_pose_in_base(
                    gripper_pose, ai_response.result_pose,
                    self.camera_config, self.hand_eye_config
                )

                # 執行抓取
                success, status_code = self._pick_at_pose(object_pose_in_base, 1.0)
                self.get_logger().info(f'debug4')
                if success:
                    self.get_logger().info(f'抓取成功，狀態碼: {status_code}')
                    self._report_success_to_ai(task_id, status_code)
                    # 是奎的位置
                    # Aruco


                    # 放置物體
                    self._place_at_pose(self.pick_place_config.default_place_pose)
                else:
                    self.get_logger().error(f'抓取失敗，狀態碼: {status_code}')
                    self._report_success_to_ai(task_id, status_code)
                self.get_logger().info(f'debug5')
                counter += 1
                time.sleep(0.5)

            # if repeat_times == -1:
            # workspace_c23322231._execute_ai_action_auto_decide(self, scenario)
            
            self.robot_controller.move_to_home()

        except Exception as e:
            self.get_logger().error(f'AI動作失敗: {e}')
            self.robot_controller.move_to_home()
            raise

