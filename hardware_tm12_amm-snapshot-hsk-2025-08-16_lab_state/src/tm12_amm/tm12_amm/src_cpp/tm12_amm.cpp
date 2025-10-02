
// abandon, complete tm12_amm.py first
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <atomic>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cctype>
#include <stdio.h>
#include <fstream>
#include <filesystem>
#include <unordered_map>

#include "eigen3/Eigen/Eigen"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "yaml-cpp/yaml.h"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_components/register_node_macro.hpp"


#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/char.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/twist.hpp"

#include "std_srvs/srv/trigger.hpp"
#include "cv_bridge/cv_bridge.h"
#include "realsense2_camera_msgs/msg/rgbd.hpp"
#include "grpr2f85_ifaces/srv/set_gripper_state.hpp"
#include "grpr2f85_ifaces/srv/get_gripper_status.hpp"
#include "tm_msgs/msg/feedback_state.hpp"
#include "tm_msgs/srv/set_positions.hpp"
#include "tm12_amm_interfaces/srv/calibration.hpp"
#include "tm12_amm/cpp_header.hpp"
#include "tm12_amm/visibility_control.h"

/*
TODO:
1. refine and check Calibration class(不要用 opencv yaml 格式)
2. 將 Calibration class 抽成獨立檔案
4. 實作 TM12_AMM class
5. 實作 TM12_AMM_Node class
*/

namespace ncku_csie_rl {};

class myCalibration
{
	public:
	myCalibration(const std::string& config_path)
	: config_path_(config_path)
	{
		// 讀取配置檔案
		if (!readConfig()) {
			throw std::runtime_error("Failed to read calibration config");
		}

		initializeObjectCorners();
	}

	~myCalibration() = default;

	bool execute()
	{
		try
		{
			validate_data();
			setTimestamp();

			if (!createDirectories() ||
			!processImages() ||
			!calculateCameraMatrix() ||
			!performHandEyeCalibration() ||
			!calculateChessboardCorners() ||
			!saveResults()) {
				return false;
			}


			return true;
		}
		catch(const std::exception& e)
		{
			std::cerr << "Exception in execute: " << e.what() << std::endl;
			return false;
		}
	}

	const std::string& getResultPath() const { return result_path_; }

	bool append_data(const cv::Mat& Img, const cv::Mat& depthImg, const std::array<double, 6>& robot_pose)
	{
		rgbImages_.push_back(Img);
		depthImages_.push_back(depthImg);
		robotPoses_.push_back(robot_pose);
		return true;
	}

	std::vector<std::array<double,6>> getTrajectory() {
		std::vector<std::array<double,6>> result;

		// Convert each vector<double> to array<double,6>
		for(const auto& traj : trajectories_) {
			std::array<double,6> arr;
			// Ensure we only copy exactly 6 elements
			std::copy(traj.begin(),
					  traj.begin() + std::min(size_t(6), traj.size()),
					  arr.begin());
			result.push_back(arr);
		}

		return result;
	}

	private:

	struct CalibrationResult
	{
		double error;
		cv::Mat cameraMatrix;
		cv::Mat distCoeffs;
		cv::Mat handEyeRotations;
		cv::Mat handEyeTranslations;
	} result_;
	std::vector<std::vector<cv::Point3d>> basePoints_;

	// 設定檔路徑
	std::filesystem::path config_path_;
	std::filesystem::path result_path_;
	std::filesystem::path image_dir_;
	std::string timestamp_;

	// 影像數據
	std::vector<cv::Mat> rgbImages_;
	std::vector<cv::Mat> depthImages_;
	std::vector<std::array<double, 6>> robotPoses_;

	// 校正相關數據
	std::vector<std::vector<cv::Point3f>> objectPoints_;
	std::vector<cv::Point3f> objectCorners_;
	std::vector<std::vector<cv::Point2f>> imagePoints_;
	std::vector<std::vector<float> > imagePointsDepth_;
	std::vector<cv::Mat> rvecsCamera_, tvecsCamera_;

	// 棋盤格參數
	int pattern_width_;
	int pattern_height_;
	double square_size_;
	std::vector<std::vector<double>> trajectories_;

	void initializeObjectCorners()
	{
		objectCorners_.clear();
		for(int i = 0; i < pattern_height_; i++) {
			for(int j = 0; j < pattern_width_; j++) {
				objectCorners_.push_back(
					cv::Point3f(j*square_size_, i*square_size_, 0.0f));
			}
		}
	}

	void setTimestamp() {
		auto now = std::chrono::system_clock::now();
		auto now_time_t = std::chrono::system_clock::to_time_t(now);
		std::stringstream ss;
		ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
		timestamp_ = ss.str();
	}

	cv::Mat getRotationFromTm12(const double& rx, const double& ry, const double& rz) {
		cv::Mat Rx = (cv::Mat_<double>(3,3) <<
		1,          0,           0,
		0, cos(rx), -sin(rx),
		0, sin(rx),  cos(rx));

		cv::Mat Ry = (cv::Mat_<double>(3,3) <<
		cos(ry), 0, sin(ry),
		0,       1,       0,
		-sin(ry), 0, cos(ry));

		cv::Mat Rz = (cv::Mat_<double>(3,3) <<
		cos(rz), -sin(rz), 0,
		sin(rz),  cos(rz), 0,
		0,        0,       1);

		return Rz * Ry * Rx;
	}

	void validate_data() {

		if (rgbImages_.empty() || depthImages_.empty() || robotPoses_.empty()) {
			throw std::runtime_error("No images or poses available for processing");
		}

		if (rgbImages_.size() != depthImages_.size() ||
		rgbImages_.size() != robotPoses_.size()) {
			throw std::runtime_error("Inconsistent data sizes");
		}
	}

	bool readConfig()
	{
		try {
			YAML::Node config = YAML::LoadFile(config_path_);
			if (!config["tm12_amm"]) {
				throw std::runtime_error("Cannot find tm12_amm node in config file");
			}

			auto tm12_config = config["tm12_amm"];
			if (!tm12_config["sampleTrajectory"] || !tm12_config["chessboard_info"]) {
				throw std::runtime_error("Missing required configuration fields");
			}

			// 讀取採樣軌跡
			trajectories_ = tm12_config["sampleTrajectory"].as<std::vector<std::vector<double>>>();

			// 讀取棋盤格資訊
			auto chessboard = tm12_config["chessboard_info"];
			if (!chessboard["pattern_width"] || !chessboard["pattern_height"] || !chessboard["square_size"]) {
				throw std::runtime_error("Missing chessboard configuration");
			}

			pattern_width_ = chessboard["pattern_width"].as<int>();
			pattern_height_ = chessboard["pattern_height"].as<int>();
			square_size_ = chessboard["square_size"].as<double>();

			return true;
		}
		catch(const std::exception& e) {
			RCLCPP_ERROR(rclcpp::get_logger("CalibrationExecutor"), "Failed to read config: %s", e.what());
			return false;
		}
	}

	// 建立目錄結構
	bool createDirectories() {
		try {
			std::filesystem::path save_dir("/workspaces/Hardware/src/ncku_csie_rl/tm12_amm/data/calibration");
			save_dir = save_dir / timestamp_;
			std::filesystem::path image_dir = save_dir / "images";

			if (!std::filesystem::exists(save_dir)) {
				std::filesystem::create_directories(save_dir);
			}
			if (!std::filesystem::exists(image_dir)) {
				std::filesystem::create_directories(image_dir);
			}

			result_path_ = save_dir.string();
			image_dir_ = image_dir.string();
			return true;
		}
		catch(const std::exception& e) {
			std::cerr << "Failed to create directories: " << e.what() << std::endl;
			return false;
		}
	}

	// 處理影像並尋找棋盤格角點
	bool processImages() {

		try {
			cv::Size patternSize(pattern_width_, pattern_height_);
			int valid_image_count = 0;

			for(size_t i = 0; i < rgbImages_.size(); i++) {
				cv::Mat gray;
				cv::cvtColor(rgbImages_[i], gray, cv::COLOR_BGR2GRAY);

				std::vector<cv::Point2f> corners;
				bool found = cv::findChessboardCorners(gray, patternSize, corners);

				if(found) {
					// 收集角點深度資訊
					std::vector<float> corner_depths;
					for(const auto& corner : corners) {
						float depth = depthImages_[i].at<float>(corner.y, corner.x);
						corner_depths.push_back(depth);
					}

					// 精細化角點位置
					cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
					cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

					// 儲存數據
					imagePoints_.push_back(corners);
					imagePointsDepth_.push_back(corner_depths);
					objectPoints_.push_back(objectCorners_);

					valid_image_count++;

					// 儲存原始圖像和處理後的圖像
					std::string image_filename = "calibration_" + std::to_string(valid_image_count);
					cv::imwrite((image_dir_ / (image_filename + "_rgb.jpg")).string(), rgbImages_[i]);
					cv::imwrite((image_dir_ / (image_filename + "_gray.jpg")).string(), gray);

					// 在圖像上繪製角點
					cv::Mat corner_img = rgbImages_[i].clone();
					cv::drawChessboardCorners(corner_img, patternSize, corners, found);
					cv::imwrite((image_dir_ / (image_filename + "_corners.jpg")).string(), corner_img);

					// 將位姿資訊儲存到 YAML 檔案
					YAML::Emitter pose_out;
					pose_out << YAML::BeginMap;
					pose_out << YAML::Key << "image_" + std::to_string(valid_image_count);
					pose_out << YAML::Value;
					pose_out << YAML::BeginMap;
					pose_out << YAML::Key << "robot_pose" << YAML::Value << YAML::Flow << YAML::BeginSeq;
					for (const auto& val : robotPoses_[i]) {
						pose_out << val;
					}
					pose_out << YAML::EndSeq;

					// 儲存角點座標和對應深度
					pose_out << YAML::Key << "num_corners" << YAML::Value << corners.size();

					pose_out << YAML::Key << "corners(u,v), z" << YAML::Value << YAML::BeginSeq;
					for(size_t i = 0; i < corners.size(); i++) {
						pose_out << YAML::Flow << YAML::BeginSeq;
						pose_out << corners[i].x << corners[i].y << corner_depths[i];
						pose_out << YAML::EndSeq;
					}
					pose_out << YAML::EndSeq;

					pose_out << YAML::EndMap;
					pose_out << YAML::EndMap;

					std::ofstream fout((image_dir_ / (image_filename + "_pose.yaml")).string());
					fout << pose_out.c_str();
					fout.close();
				}
			}

			return !imagePoints_.empty();
		}
		catch(const std::exception& e) {
			std::cerr << "Failed to process images: " << e.what() << std::endl;
			return false;
		}
	}

	// 計算相機矩陣和畸變係數
	bool calculateCameraMatrix() {
		try {
			cv::Size imageSize = rgbImages_[0].size();

			result_.error = cv::calibrateCamera(objectPoints_, imagePoints_, imageSize,
				result_.cameraMatrix, result_.distCoeffs, rvecsCamera_, tvecsCamera_,
                cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 | cv::CALIB_ZERO_TANGENT_DIST);
				std::cout << "Camera calibration error: " << result_.error << std::endl;

				// 計算最佳的新相機矩陣
				//newCameraMatrix_ = cv::getOptimalNewCameraMatrix(cameraMatrix_, distCoeffs_,
				//                                               imageSize, 0, imageSize);
				std::cout << "Camera matrix: " << result_.cameraMatrix << std::endl;
				std::cout << "Distortion coefficients: " << result_.distCoeffs << std::endl;
				return true;
			}
			catch(const std::exception& e) {
				std::cerr << "Failed to calculate camera matrix: " << e.what() << std::endl;
				return false;
			}
		}

	// 執行手眼校正
	bool performHandEyeCalibration() {
		try {
			std::vector<cv::Mat> camera_rotations;
			std::vector<cv::Mat> camera_translations;
			std::vector<cv::Mat> robot_rotations;
			std::vector<cv::Mat> robot_translations;

			// 檢查數據是否為空
			if (rvecsCamera_.empty() || tvecsCamera_.empty() || robotPoses_.empty()) {
			    throw std::runtime_error("Empty calibration data");
			}

			// 確保所有向量大小一致
			size_t n = std::min({rvecsCamera_.size(), tvecsCamera_.size(), robotPoses_.size()});

			// 準備相機姿態數據
			for(size_t i = 0; i < n; i++) {
			    cv::Mat R;
			    cv::Rodrigues(rvecsCamera_[i], R);
			    camera_rotations.push_back(R.clone());  // 確保深度拷貝
			    camera_translations.push_back(tvecsCamera_[i].clone());  // 確保深度拷貝
			}

			// 準備機器人姿態數據
			for(size_t i = 0; i < n; i++) {
			    cv::Mat R = getRotationFromTm12(robotPoses_[i][3],
							  robotPoses_[i][4],
							  robotPoses_[i][5]);
			    cv::Mat t = (cv::Mat_<double>(3,1) <<
				robotPoses_[i][0],
				robotPoses_[i][1],
				robotPoses_[i][2]);

			    robot_rotations.push_back(R.clone());  // 確保深度拷貝
			    robot_translations.push_back(t.clone());  // 確保深度拷貝
			}

			// 驗證輸入數據
			if (robot_rotations.size() != robot_translations.size() ||
			    camera_rotations.size() != camera_translations.size() ||
			    robot_rotations.size() != camera_rotations.size()) {
			    throw std::runtime_error("Inconsistent number of rotations and translations");
			}

			// 打印數據大小以進行調試
			RCLCPP_INFO(rclcpp::get_logger("Calibration"),
			    "Number of poses: robot_R=%zu, robot_t=%zu, camera_R=%zu, camera_t=%zu",
			    robot_rotations.size(), robot_translations.size(),
			    camera_rotations.size(), camera_translations.size());

			// 執行手眼校正
			cv::calibrateHandEye(
			    robot_rotations,      // 機器人旋轉矩陣向量
			    robot_translations,   // 機器人平移向量向量
			    camera_rotations,     // 相機旋轉矩陣向量
			    camera_translations,  // 相機平移向量向量
			    result_.handEyeRotations,    // 輸出的手眼校正轉換矩陣
			    result_.handEyeTranslations, // 輸出的手眼校正平移向量
			    cv::CALIB_HAND_EYE_TSAI    // 使用 Tsai 方法
			);

			return true;		}
		catch(const std::exception& e) {
			std::cerr << "Failed to perform hand-eye calibration: " << e.what() << std::endl;
			return false;
		}
	}

	// 計算棋盤格角點在基座標系中的位置
	bool calculateChessboardCorners() {
		try {
			for(size_t frame = 0; frame < imagePoints_.size(); frame++) {
				std::vector<cv::Point3d> camera_points;

				// 反投影到相機座標系
				for(size_t i = 0; i < imagePoints_[frame].size(); i++) {
					double depth = imagePointsDepth_[frame][i];
					if(depth <= 0) continue;

					double x = (imagePoints_[frame][i].x - result_.cameraMatrix.at<double>(0,2)) / result_.cameraMatrix.at<double>(0,0);
					double y = (imagePoints_[frame][i].y - result_.cameraMatrix.at<double>(1,2)) / result_.cameraMatrix.at<double>(1,1);

					camera_points.push_back(cv::Point3d(x * depth, y * depth, depth));
				}

				// 轉換到基座標系
				cv::Mat T_cam2gripper = cv::Mat::eye(4, 4, CV_64F);
				result_.handEyeRotations.copyTo(T_cam2gripper(cv::Rect(0,0,3,3)));
				result_.handEyeTranslations.copyTo(T_cam2gripper(cv::Rect(3,0,1,3)));

				cv::Mat T_gripper2base = cv::Mat::eye(4, 4, CV_64F);
				auto pose = robotPoses_[frame];
				cv::Mat R = getRotationFromTm12(pose[3], pose[4], pose[5]);
				R.copyTo(T_gripper2base(cv::Rect(0,0,3,3)));
				T_gripper2base.at<double>(0,3) = pose[0];
				T_gripper2base.at<double>(1,3) = pose[1];
				T_gripper2base.at<double>(2,3) = pose[2];

				cv::Mat T_total = T_gripper2base * T_cam2gripper;
				std::vector<cv::Point3d> basePointsFrame_;

				// 轉換所有點
				for(const auto& pt : camera_points) {
					cv::Mat pt_cam = (cv::Mat_<double>(4,1) << pt.x, pt.y, pt.z, 1);
					cv::Mat pt_base = T_total * pt_cam;
					basePointsFrame_.push_back(cv::Point3d(
						pt_base.at<double>(0)/pt_base.at<double>(3),
						pt_base.at<double>(1)/pt_base.at<double>(3),
						pt_base.at<double>(2)/pt_base.at<double>(3)
					));
				}
				basePoints_.push_back(basePointsFrame_);
			}

			return true;
		}
		catch(const std::exception& e) {
			std::cerr << "Failed to calculate chessboard corners: " << e.what() << std::endl;
			return false;
		}
	}

	// 儲存校正結果
	bool saveResults() {
		try {
			// 儲存校正結果
			std::filesystem::path result_file = result_path_ / "calibration_result.yaml";

			// 將 cv::Mat 轉換為標準的數據格式
			std::vector<double> camera_matrix_data;
			camera_matrix_data.assign((double*)result_.cameraMatrix.datastart,
			(double*)result_.cameraMatrix.dataend);

			std::vector<double> dist_coeffs_data;
			dist_coeffs_data.assign((double*)result_.distCoeffs.datastart,
			(double*)result_.distCoeffs.dataend);

			std::vector<double> hand_eye_rotations_data;
			hand_eye_rotations_data.assign((double*)result_.handEyeRotations.datastart,
			(double*)result_.handEyeRotations.dataend);

			std::vector<double> hand_eye_translations_data;
			hand_eye_translations_data.assign((double*)result_.handEyeTranslations.datastart,
			(double*)result_.handEyeTranslations.dataend);

			// 儲存校正結果
			YAML::Emitter calib_out;
			calib_out << YAML::BeginMap;

			// 儲存校正誤差
			calib_out << YAML::Key << "error" << YAML::Value << result_.error;

			// 儲存相機矩陣 (3x3)
			calib_out << YAML::Key << "camera_matrix" << YAML::Value;
			calib_out << YAML::BeginSeq;
			for (int i = 0; i < 3; i++) {
				calib_out << YAML::Flow << YAML::BeginSeq;
				for (int j = 0; j < 3; j++) {
					calib_out << camera_matrix_data[i * 3 + j];
				}
				calib_out << YAML::EndSeq;
			}
			calib_out << YAML::EndSeq;

			// 儲存畸變係數
			calib_out << YAML::Key << "distortion_coefficients" << YAML::Value;
			calib_out << YAML::Flow << YAML::BeginSeq;
			for (const auto& coeff : dist_coeffs_data) {
				calib_out << coeff;
			}
			calib_out << YAML::EndSeq;

			// 儲存手眼校正旋轉矩陣 (3x3)
			calib_out << YAML::Key << "hand_eye_rotation" << YAML::Value;
			calib_out << YAML::BeginSeq;
			for (int i = 0; i < 3; i++) {
				calib_out << YAML::Flow << YAML::BeginSeq;
				for (int j = 0; j < 3; j++) {
					calib_out << hand_eye_rotations_data[i * 3 + j];
				}
				calib_out << YAML::EndSeq;
			}
			calib_out << YAML::EndSeq;

			// 儲存手眼校正平移向量
			calib_out << YAML::Key << "hand_eye_translation" << YAML::Value;
			calib_out << YAML::Flow << YAML::BeginSeq;
			for (const auto& trans : hand_eye_translations_data) {
				calib_out << trans;
			}
			calib_out << YAML::EndSeq;

			calib_out << YAML::EndMap;

			// 寫入校正結果文件
			std::ofstream calib_fout(result_file.string());
			calib_fout << calib_out.c_str();
			calib_fout.close();

			// 儲存角點位置數據
			std::filesystem::path corners_file = result_path_ / "corners_in_base_frame.yaml";
			YAML::Emitter corners_out;
			corners_out << YAML::BeginMap;
			corners_out << YAML::Key << "corners_in_base_frame" << YAML::Value << YAML::BeginSeq;

			// 為每一幀儲存角點位置
			for(size_t frame = 0; frame < basePoints_.size(); frame++) {
				corners_out << YAML::BeginMap;
				corners_out << YAML::Key << "frame_" + std::to_string(frame + 1);
				corners_out << YAML::Value << YAML::BeginSeq;

				// 儲存該幀中的所有角點
				for(const auto& corner : basePoints_[frame]) {
					corners_out << YAML::Flow << YAML::BeginSeq;
					corners_out << corner.x << corner.y << corner.z;
					corners_out << YAML::EndSeq;
				}

				corners_out << YAML::EndSeq;
				corners_out << YAML::EndMap;
			}

			corners_out << YAML::EndSeq;
			corners_out << YAML::EndMap;

			// 寫入角點位置文件
			std::ofstream corners_fout(corners_file.string());
			corners_fout << corners_out.c_str();
			corners_fout.close();

			return true;
		}
		catch(const std::exception& e) {
			std::cerr << "Failed to save results: " << e.what() << std::endl;
			return false;
		}
	}

};

class TM12_AMM_old : public rclcpp::Node {
	public:
	TM12_AMM_old() : Node("tm12_amm") {
		try
		{
			clt_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
			srv_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
			sub_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
			sub_options.callback_group = sub_cb_group_;
			declareParameters();
			setupPublishers();
			setupSubscriptions();
			setupClients();
			setupServices();

			//topHomingExecute();
		}
		catch(const std::exception& e)
		{
			RCLCPP_ERROR(this->get_logger(), "Exception in TM12_AMM constructor: %s", e.what());
		}
	}

	~TM12_AMM_old() = default;

	private:
	using Char = std_msgs::msg::Char;
	using Twist = geometry_msgs::msg::Twist;
	using Img = sensor_msgs::msg::Image;
	using TM_Pose = tm_msgs::srv::SetPositions;
	using TM_FeedbackState = tm_msgs::msg::FeedbackState;
	using Trigger = std_srvs::srv::Trigger;
	using GRPR_2f85_Set = grpr2f85_ifaces::srv::SetGripperState;
	using GRPR_2f85_Get = grpr2f85_ifaces::srv::GetGripperStatus;
	using Calibration = tm12_amm_interfaces::srv::Calibration;

	double z_min_;
	double velocity_max_;
	double acc_time_min_;
	std::array<double, 6> home_in_joint_;

	bool manual_mode_;
	cv::Mat camera_matrix_;
	cv::Mat dist_coeffs_;
	cv::Mat new_camera_matrix_;
	cv::Mat mapx_;
	cv::Mat mapy_;
	cv::Mat R_c2g_;
	cv::Mat t_c2g_;
	std::array<double, 6> pose_take_photo_;


	rclcpp::CallbackGroup::SharedPtr clt_cb_group_;
	rclcpp::CallbackGroup::SharedPtr srv_cb_group_;
	rclcpp::CallbackGroup::SharedPtr sub_cb_group_;
	rclcpp::SubscriptionOptions sub_options;


	rclcpp::TimerBase::SharedPtr timer_;

	rclcpp::Subscription<Img>::SharedPtr rgb_realsense_subscription_;
	rclcpp::Subscription<Img>::SharedPtr depth_realsense_subscription_;
	cv::Mat rgb_image_, depth_image_;

	rclcpp::Publisher<Img>::SharedPtr calibrated_rgb_publisher_;
	rclcpp::Publisher<Img>::SharedPtr calibrated_depth_publisher_;

	rclcpp::Client<GRPR_2f85_Set>::SharedPtr grpr2f85_set_gripper_state_client_;
	rclcpp::Client<GRPR_2f85_Get>::SharedPtr grpr2f85_get_gripper_state_client_;


	rclcpp::Publisher<Twist>::SharedPtr amr_twist_publisher_;
	rclcpp::Client<Trigger>::SharedPtr amr_servo_on_client_;
	rclcpp::Client<Trigger>::SharedPtr amr_servo_off_client_;

	rclcpp::Subscription<TM_FeedbackState>::SharedPtr tm12_feedback_subscription_;
	TM_FeedbackState::SharedPtr tm12_feedback_state_;
	rclcpp::Client<TM_Pose>::SharedPtr tm12_set_positions_client_;

	rclcpp::Service<Trigger>::SharedPtr homing_;

	rclcpp::Subscription<Char>::SharedPtr keyboard_manual_subscription_;
	OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

	rclcpp::Service<Calibration>::SharedPtr calibration_;
	rclcpp::Service<Trigger>::SharedPtr verify_calibration_;

	/*
	calibration, verify_calibration, do_task to action
	*/



	////////////////////////////////////////////////////////////////////////////////////////////////////////

	void declareParameters()
    {
		try
		{
			rcl_interfaces::msg::ParameterDescriptor read_only;
			read_only.read_only = true;

			z_min_ = this->declare_parameter<double>("z_min", 0.02, read_only);
			velocity_max_ = this->declare_parameter<double>("velocity_max", 2.0, read_only);
			acc_time_min_ = this->declare_parameter<double>("acc_time_min", 0.2, read_only);
			std::vector<double> home_in_joint_tmp = this->declare_parameter<std::vector<double>>("home_in_joint",
				{-M_PI_4, 0.0, M_PI_2, 0.0, M_PI_2, 0.0}, read_only);
			std::copy(home_in_joint_tmp.begin(), home_in_joint_tmp.end(), home_in_joint_.begin());

			manual_mode_ = this->declare_parameter<bool>("manual_mode", 1);

            // 宣告參數並設定預設值
            std::vector<double> matrix = this->declare_parameter<std::vector<double>>("camera.matrix", {});
            std::vector<double> distortion = this->declare_parameter<std::vector<double>>("camera.distortion", {});
            std::vector<double> rotation = this->declare_parameter<std::vector<double>>("hand_eye.rotation", {});
            std::vector<double> translation = this->declare_parameter<std::vector<double>>("hand_eye.translation", {});


            // 轉換成 OpenCV Mat 格式
            camera_matrix_ = cv::Mat(matrix).reshape(1, 3);
            dist_coeffs_ = cv::Mat(distortion).reshape(1, 1);
            R_c2g_ = cv::Mat(rotation).reshape(1, 3);
            t_c2g_ = cv::Mat(translation).reshape(1, 3);

		std::vector<double> pose_take_photo_tmp = this->declare_parameter<std::vector<double>>("pose_take_photo",
					{0.345, -0.518, 0.4, -M_PI, 0.0, M_PI_4});
		std::copy(pose_take_photo_tmp.begin(), pose_take_photo_tmp.end(), pose_take_photo_.begin());

		parameter_callback_handle_ = this->add_on_set_parameters_callback(
				std::bind(&TM12_AMM_old::parameter_callback, this, std::placeholders::_1));

		}
        catch (const std::exception& e)
        {
			RCLCPP_ERROR(this->get_logger(), "Exception in declareParameters: %s", e.what());
		}
	}

	void setupSubscriptions() {
		try {
			rgb_realsense_subscription_ = this->create_subscription<Img>(
				"/camera/realsense_camera/color/image_raw",
				10,
				[this](const Img::SharedPtr msg) {
					try {

						rgb_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image.clone();
					} catch (cv_bridge::Exception& e) {
						RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
					}
				},
				sub_options
			);

			depth_realsense_subscription_ = this->create_subscription<Img>(
				"/camera/realsense_camera/aligned_depth_to_color/image_raw",
				10,
				[this](const Img::SharedPtr msg){
					try {
						depth_image_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image.clone();
						depth_image_/=1000.0;
					} catch (cv_bridge::Exception& e) {
						RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
					}
				},
				sub_options
			);


			tm12_feedback_subscription_ = this->create_subscription<TM_FeedbackState>(
				"feedback_states", 10,
				[this](const TM_FeedbackState::SharedPtr msg){
					try {
						tm12_feedback_state_ = msg;
					} catch (const std::exception& e) {
						RCLCPP_ERROR(this->get_logger(), "Exception in tm12FeedbackCallback: %s", e.what());
					}
				},
				sub_options
			);

			if(true) {
				keyboard_manual_subscription_ = this->create_subscription<Char>(
					"keyboard_manual", 0,
					std::bind(&TM12_AMM_old::keyboardManualCallback, this, std::placeholders::_1)
				);
			}

		} catch (const std::exception& e) {
			RCLCPP_ERROR(this->get_logger(), "Exception in setupSubscriptions: %s", e.what());
		}
	}

	void setupPublishers(){
		try {
			amr_twist_publisher_ = this->create_publisher<Twist>(
				"amr_twist", 10
			);

			timer_ = this->create_wall_timer(std::chrono::milliseconds(100),
			std::bind(&TM12_AMM_old::timerCallback, this));

		} catch (const std::exception& e) {
			RCLCPP_ERROR(this->get_logger(), "Exception in setupPublishers: %s", e.what());
		}
	}

	void timerCallback() {
		try {
			// 檢查影像是否為空
			if (rgb_image_.empty() || depth_image_.empty()) {
				return;
			}

			// 2. 建立 ROS 標頭
			std_msgs::msg::Header header;
			header.stamp = this->now();
			header.frame_id = "camera_frame"; // 設定座標系

		} catch (const std::exception& e) {
			RCLCPP_ERROR(this->get_logger(), "Exception in timerCallback: %s", e.what());
		}
	}

	void setupClients() {
		try {
			std::thread thread1([this]() {
				tm12_set_positions_client_ = createClientAndWait<TM_Pose>("set_positions");
			});
			std::thread thread2([this]() {
				grpr2f85_set_gripper_state_client_ = createClientAndWait<GRPR_2f85_Set>("set_gripper_state");
			});
			std::thread thread3([this]() {
				grpr2f85_get_gripper_state_client_ = createClientAndWait<GRPR_2f85_Get>("get_gripper_status");
			});
			std::thread thread4([this]() {
				amr_servo_on_client_ = createClientAndWait<Trigger>("servo_on");
			});
			std::thread thread5([this]() {
				amr_servo_off_client_ = createClientAndWait<Trigger>("servo_off");
			});
			thread1.join();
			thread2.join();
			thread3.join();
			thread4.join();
			thread5.join();
		} catch (const std::exception& e) {
			RCLCPP_ERROR(this->get_logger(), "Exception in setupClients: %s", e.what());
		}
	}

	void setupServices() {
		try {
			homing_ = this->create_service<Trigger>(
				"homing",
				[this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> res) {
					handleHomingRequest(req, res);
				},
				rmw_qos_profile_services_default,
				srv_cb_group_
			);
			calibration_ = this->create_service<Calibration>(
				"calibration",
				[this](const std::shared_ptr<Calibration::Request> req, std::shared_ptr<Calibration::Response> res) {
					handleCalibrationRequest(req, res);
				},
				rmw_qos_profile_services_default,
				srv_cb_group_
			);

			verify_calibration_ = this->create_service<Trigger>(
				"verify_calibration",
				[this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> res) {
					handleVerifyCalibrationRequest(req, res);
				},
				rmw_qos_profile_services_default,
				srv_cb_group_
			);

		} catch (const std::exception& e) {
			RCLCPP_ERROR(this->get_logger(), "Exception in setupServices: %s", e.what());
		}
	}

	cv::Mat get_R_from_tm12(double rx, double ry, double rz)
	{

		// 歐拉角轉旋轉矩陣 (EulerZYX)
		// 建立旋轉矩陣 R = Rz * Ry * Rx
		cv::Mat Rx = (cv::Mat_<double>(3,3) <<
		1,          0,           0,
		0, cos(rx), -sin(rx),
		0, sin(rx),  cos(rx));

		cv::Mat Ry = (cv::Mat_<double>(3,3) <<
		cos(ry), 0, sin(ry),
		0,       1,       0,
		-sin(ry), 0, cos(ry));

		cv::Mat Rz = (cv::Mat_<double>(3,3) <<
		cos(rz), -sin(rz), 0,
		sin(rz),  cos(rz), 0,
		0,        0,       1);

		return Rz * Ry * Rx;
	}

	void Tranformaton_C2B()
	{

	}



				////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename ServiceT>
	typename rclcpp::Client<ServiceT>::SharedPtr createClientAndWait(const std::string & service_name, int max_attempts = 10) {
		auto client = this->create_client<ServiceT>(service_name, rmw_qos_profile_services_default,
			clt_cb_group_);
			int attempts = 0;
			do {
				if (client->wait_for_service(std::chrono::seconds(5))) {
					RCLCPP_INFO(this->get_logger(), "Service %s is now available.", service_name.c_str());
					return client;
				}
				RCLCPP_ERROR(this->get_logger(), "Attempts: %d, %s service server isn't ready. Please check whether the server is alive", attempts + 1, service_name.c_str());
				std::this_thread::sleep_for(std::chrono::seconds(1));
				attempts++;
			} while (rclcpp::ok() && attempts < max_attempts);

			RCLCPP_ERROR(this->get_logger(), "Max attempts reached. Service %s is not available.", service_name.c_str());
			rclcpp::shutdown();  // 強制中止節點
			return nullptr;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////////

	rclcpp::Client<TM_Pose>::SharedFuture callTMDriverSetPosition(const std::array<double, 6>& position = {0.3571, -0.5795, 0.5, -M_PI, 0.0, M_PI_4},
		int motion_type = 2, double velocity = 1.0, double acc_time = 1.0,
		int blend_percentage = 0, bool fine_goal = true, int wait_time = 0)
		{
			auto req = std::make_shared<TM_Pose::Request>();
			if (motion_type == 1 || motion_type == 2 || motion_type == 4) {
				req->motion_type = motion_type;
			} else {
				throw std::invalid_argument("Motion_type is not allowed: PTP_J = 1, PTP_T = 2, LINE_T = 4");
			}
			std::array<double, 6> checked_positions_array = rangeCheckCartesian(position, motion_type);
			req->positions.assign(checked_positions_array.begin(), checked_positions_array.end());
			req->velocity = std::min(velocity, this->get_parameter("velocity_max").as_double());
			req->acc_time = std::max(acc_time, this->get_parameter("acc_time_min").as_double());
			req->blend_percentage = std::clamp(blend_percentage, 0, 100);
			req->fine_goal = fine_goal;

			RCLCPP_INFO(this->get_logger(), "Calling tm_driver set_position service");
			auto future = tm12_set_positions_client_->async_send_request(req);

			std::future_status status = future.wait_for(std::chrono::seconds(wait_time));

			return future;
		}

	std::array<double, 6> rangeCheckCartesian(const std::array<double, 6>& position, const int& motion_type) {
		std::array<double, 6> ans(position);
		if (motion_type == 2) {
			ans.at(2) = std::max(ans.at(2), this->get_parameter("z_min").as_double());
		}
		return ans;
	}

	rclcpp::Client<GRPR_2f85_Set>::SharedFuture callGrpr2f85SetState(int position = 0, int speed = 255, int force = 255, int wait_time = 0) {
		auto req = std::make_shared<GRPR_2f85_Set::Request>();
		req->position = std::clamp(position, 0, 255);
		req->speed = speed;
		req->force = force;
		req->wait_time = wait_time;

		RCLCPP_INFO(this->get_logger(), "Calling gripper set state service");
		auto future = grpr2f85_set_gripper_state_client_->async_send_request(req);

		//future.wait();

		return future;
	}

	rclcpp::Client<GRPR_2f85_Get>::SharedFuture callGrpr2f85GetStatus() {
		auto req = std::make_shared<GRPR_2f85_Get::Request>();
		RCLCPP_INFO(this->get_logger(), "Calling gripper get state service");
		auto future = grpr2f85_get_gripper_state_client_->async_send_request(req);

		//future.wait();

		return future;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////

	bool topHomingExecute() {
		RCLCPP_INFO(this->get_logger(), "Both TM12 & Gripper homing");
		auto home_in_joint = this->get_parameter("home_in_joint").as_double_array();
		std::array<double, 6> home_in_joint_array;
		std::copy_n(home_in_joint.begin(), 6, home_in_joint_array.begin());
		auto tm_position_result = callTMDriverSetPosition(home_in_joint_array, 1).get();
		auto gripper_state_result = callGrpr2f85SetState().get();

		if (tm_position_result->ok && gripper_state_result->ok) {
			RCLCPP_INFO(this->get_logger(), "Both TM12 & Gripper are at home");
			return true;
		}
		RCLCPP_INFO(this->get_logger(), "Homing failed");
		return false;
	}

	void handleHomingRequest(const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> res) {
		try {
				res->success = topHomingExecute();
		} catch (const std::exception& e) {
			RCLCPP_ERROR(this->get_logger(), "Exception in handleHomingRequest: %s", e.what());
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////

	void create_subscriber()
	{
		if (!keyboard_manual_subscription_)
		{
			RCLCPP_INFO(this->get_logger(), "Creating subscriber.");
			keyboard_manual_subscription_ = this->create_subscription<Char>(
				"topic_name", 10, std::bind(&TM12_AMM_old::keyboardManualCallback, this, std::placeholders::_1));
			}
		}

	void destroy_subscriber()
	{
		if (keyboard_manual_subscription_)
		{
			RCLCPP_INFO(this->get_logger(), "Destroying subscriber.");
			keyboard_manual_subscription_.reset();
		}
	}

	rcl_interfaces::msg::SetParametersResult parameter_callback(const std::vector<rclcpp::Parameter> &parameters)
	{
		rcl_interfaces::msg::SetParametersResult result;
		result.successful = true;

		for (const auto &param : parameters)
		{
			if (param.get_name() == "manual_mode")
			{
				if (param.as_bool())
				{
					create_subscriber();
				}
				else
				{
					destroy_subscriber();
				}
			}
		}
		return result;
	}

	void keyboardManualCallback(const Char::SharedPtr msg) {
		try {
			std::array<double, 6> pose_next;
			std::copy(tm12_feedback_state_->tool_pose.begin(), tm12_feedback_state_->tool_pose.end(), pose_next.begin());
			handleKeyPress(msg->data, pose_next);
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
		} catch (const std::exception& e) {
			RCLCPP_ERROR(this->get_logger(), "Exception in keyboardManualCallback: %s", e.what());
		}
	}

	void handleKeyPress(int ch, std::array<double, 6>& pose_next, double speed = 5) {
		try {

			Twist twist_msg;
			switch (ch) {
				case 'q':
				RCLCPP_INFO(this->get_logger(), "+x");
				pose_next.at(0) += speed * 0.001 * std::cos(-M_PI_4);
				pose_next.at(1) += speed * 0.001 * std::sin(-M_PI_4);
				callTMDriverSetPosition(pose_next);
				break;
				case 'a':
				RCLCPP_INFO(this->get_logger(), "-x");
				pose_next.at(0) -= speed * 0.001 * std::cos(-M_PI_4);
				pose_next.at(1) -= speed * 0.001 * std::sin(-M_PI_4);
				callTMDriverSetPosition(pose_next);
				break;
				case 'w':
				RCLCPP_INFO(this->get_logger(), "+y");
				pose_next.at(0) += speed * 0.001 * -std::sin(-M_PI_4);
				pose_next.at(1) += speed * 0.001 * std::cos(-M_PI_4);
				callTMDriverSetPosition(pose_next);
				break;
				case 's':
				RCLCPP_INFO(this->get_logger(), "-y");
				pose_next.at(0) -= speed * 0.001 * -std::sin(-M_PI_4);
				pose_next.at(1) -= speed * 0.001 * std::cos(-M_PI_4);
				callTMDriverSetPosition(pose_next);
				break;
				case 'e':
				RCLCPP_INFO(this->get_logger(), "+z");
				pose_next.at(2) += speed * 0.001;
				callTMDriverSetPosition(pose_next);
				break;
				case 'd':
				RCLCPP_INFO(this->get_logger(), "-z");
				pose_next.at(2) -= speed * 0.001;
				callTMDriverSetPosition(pose_next);
				break;
				case 'r':
				RCLCPP_INFO(this->get_logger(), "+roll");
				pose_next.at(3) += speed * M_PI / 1800;
				callTMDriverSetPosition(pose_next);
				break;
				case 'f':
				RCLCPP_INFO(this->get_logger(), "-roll");
				pose_next.at(3) -= speed * M_PI / 1800;
				callTMDriverSetPosition(pose_next);
				break;
				case 't':
				RCLCPP_INFO(this->get_logger(), "+pitch");
				pose_next.at(4) += speed * M_PI / 1800;
				callTMDriverSetPosition(pose_next);
				break;
				case 'g':
				RCLCPP_INFO(this->get_logger(), "-pitch");
				pose_next.at(4) -= speed * M_PI / 1800;
				callTMDriverSetPosition(pose_next);
				break;
				case 'y':
				RCLCPP_INFO(this->get_logger(), "+yaw");
				pose_next.at(5) += speed * M_PI / 1800;
				callTMDriverSetPosition(pose_next);
				break;
				case 'h':
				RCLCPP_INFO(this->get_logger(), "-yaw");
				pose_next.at(5) -= speed * M_PI / 1800;
				callTMDriverSetPosition(pose_next);
				break;
				case 'z':
				RCLCPP_INFO(this->get_logger(), "Gripper close");
				callGrpr2f85SetState(255);
				break;
				case 'x':
				RCLCPP_INFO(this->get_logger(), "Gripper open");
				callGrpr2f85SetState(0);
				break;
				case 'c':
				RCLCPP_INFO(this->get_logger(), "AMR linear velocity(+)");
				//Twist twist_msg;
				twist_msg.linear.x = 0.1;
				twist_msg.angular.z = 0.0;
				amr_twist_publisher_->publish(twist_msg);
				break;
				case 'v':
				RCLCPP_INFO(this->get_logger(), "AMR linear velocity(-)");
				//Twist twist_msg;
				twist_msg.linear.x = -0.1;
				twist_msg.angular.z = 0.0;
				amr_twist_publisher_->publish(twist_msg);
				break;
				case 'b':
				RCLCPP_INFO(this->get_logger(), "AMR angular velocity(+)");
				//Twist twist_msg;
				twist_msg.linear.x = 0.0;
				twist_msg.angular.z = 0.1;
				amr_twist_publisher_->publish(twist_msg);
				break;
				case 'n':
				RCLCPP_INFO(this->get_logger(), "AMR angular velocity(-)");
				//Twist twist_msg;
				twist_msg.linear.x = 0.0;
				twist_msg.angular.z = -0.1;
				amr_twist_publisher_->publish(twist_msg);
				break;
				case 27:
				topHomingExecute();
				break;
				default:
				break;
			}
		} catch (const std::exception& e) {
			RCLCPP_ERROR(this->get_logger(), "Exception in handleKeyPress: %s", e.what());
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////

	void handleCalibrationRequest(const std::shared_ptr<Calibration::Request> req, std::shared_ptr<Calibration::Response> res) {
		try {
			//calibrationExecute(req, res);
			calibrationExecute_v2(req, res);
		} catch (const std::exception& e) {
			RCLCPP_ERROR(this->get_logger(), "Exception in handleCalibrationRequest: %s", e.what());
		}
	}

	void calibrationExecute(const std::shared_ptr<Calibration::Request>& req, std::shared_ptr<Calibration::Response>& res) {
		RCLCPP_INFO(this->get_logger(), "Calibration");
		try
		{
			// 1. Initial

			// 確保目標目錄存在
			std::filesystem::path save_dir("/workspaces/Hardware/src/ncku_csie_rl/tm12_amm/data/calibration");

			auto now = std::chrono::system_clock::now();
			auto now_time_t = std::chrono::system_clock::to_time_t(now);
			std::stringstream ss;
			ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
			std::string timestamp = ss.str();

			save_dir = save_dir / timestamp;

			std::filesystem::path image_dir = save_dir / "images";

			if (!std::filesystem::exists(save_dir)) {
				std::filesystem::create_directories(save_dir);
			}
			if (!std::filesystem::exists(image_dir)) {
				std::filesystem::create_directories(image_dir);
			}

			std::vector<std::vector<cv::Point3f>> objectPoints;
			std::vector<std::vector<cv::Point2f>> imagePoints;
			std::vector<std::vector<float> > imagePointsDepth;
			std::vector<cv::Mat> rvecsCamera, tvecsCamera;
			std::vector<cv::Mat> robot_poses;

			// 2. Read calibration config file
			// 修正 YAML 設定檔讀取部分
			// 2. Read calibration config file
			YAML::Node config = YAML::LoadFile("/workspaces/Hardware/src/ncku_csie_rl/tm12_amm/config/calibration_camera_eih_001.yaml");
			if (!config["tm12_amm"]) {
				throw std::runtime_error("Cannot find tm12_amm node in config file");
			}

			auto tm12_config = config["tm12_amm"];
			if (!tm12_config["sampleTrajectory"] || !tm12_config["chessboard_info"]) {
				throw std::runtime_error("Missing required configuration fields");
			}

			// 讀取採樣軌跡
			auto trajectories = tm12_config["sampleTrajectory"].as<std::vector<std::vector<double>>>();

			// 讀取棋盤格資訊
			auto chessboard = tm12_config["chessboard_info"];
			if (!chessboard["pattern_width"] || !chessboard["pattern_height"] || !chessboard["square_size"]) {
				throw std::runtime_error("Missing chessboard configuration");
			}

			int pattern_width = chessboard["pattern_width"].as<int>();
			int pattern_height = chessboard["pattern_height"].as<int>();  // 修正拼字錯誤 "pattern_heigh" -> "pattern_height"
			double squareSize = chessboard["square_size"].as<double>();

			// 建立棋盤格大小
			cv::Size patternSize(pattern_width, pattern_height);

			RCLCPP_INFO(this->get_logger(), "Calibration config loaded: pattern_size=(%d,%d), square_size=%.3f",
			pattern_width, pattern_height, squareSize);

			std::vector<cv::Point3f> objectCorners;
			for(int i = 0; i < patternSize.height; i++) {
				for(int j = 0; j < patternSize.width; j++) {
					objectCorners.push_back(cv::Point3f(j*squareSize, i*squareSize, 0.0f));
				}
			}

			// 3. Move and take photo
			int valid_image_count = 0;
			for (const auto& pose : trajectories)
			{
				std::array<double, 6> robot_pose;
				for(size_t i = 0; i < 6; i++) {
					robot_pose[i] = pose[i];
				}
				auto result = callTMDriverSetPosition(robot_pose, 2,
					1.0, 1.0, 100, 1, 1).get();
					if(!result->ok) {
						RCLCPP_ERROR(this->get_logger(), "Failed to move robot to calibration pose");
						continue;
					}
					// 等待機器人到達位置和相機穩定
					std::this_thread::sleep_for(std::chrono::seconds(3));

					std::array<double, 6> actual_pose;
					std::copy(tm12_feedback_state_->tool_pose.begin(),
					tm12_feedback_state_->tool_pose.end(),
					actual_pose.begin());

					// 儲存相機圖像
					cv::Mat rgb = rgb_image_.clone(),
					depth = depth_image_.clone(),
					gray;
					cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

					// 尋找棋盤格角點
					std::vector<cv::Point2f> corners;
					bool found = cv::findChessboardCorners(gray, patternSize, corners);

					if(found) {
						valid_image_count++;

						// 儲存原始圖像和處理後的圖像
						std::string image_filename = "calibration_" + std::to_string(valid_image_count);
						cv::imwrite((image_dir / (image_filename + "_rgb.jpg")).string(), rgb);
						cv::imwrite((image_dir / (image_filename + "_gray.jpg")).string(), gray);

						// 在圖像上繪製角點
						cv::Mat corner_img = rgb.clone();
						cv::drawChessboardCorners(corner_img, patternSize, corners, found);
						cv::imwrite((image_dir / (image_filename + "_corners.jpg")).string(), corner_img);

						std::vector<float> corner_depths;
						for(const auto& corner : corners) {
							// 從深度圖像中取得深度值(單位:米)
							float depth_value = depth.at<float>(corner.y, corner.x);

							corner_depths.push_back(static_cast<float>(depth_value));
						}

						// 精細化角點位置
						cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
						cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

						// 儲存找到的角點
						imagePoints.push_back(corners);
						objectPoints.push_back(objectCorners);
						imagePointsDepth.push_back(corner_depths);
						// 使用實際回饋的機器人位姿
						cv::Mat robot_transform = cv::Mat::eye(4, 4, CV_64F);
						// 填充旋轉和平移
						for(int i = 0; i < 3; i++) {
							robot_transform.at<double>(i,3) = actual_pose[i];
						}

						// 歐拉角轉旋轉矩陣 (EulerZYX)
						//cv::Mat R;
						//double rx = actual_pose[3]; // roll (X)
						//double ry = actual_pose[4]; // pitch (Y)
						//double rz = actual_pose[5]; // yaw (Z)

						// 建立旋轉矩陣 R = Rz * Ry * Rx
						//cv::Mat Rx = (cv::Mat_<double>(3,3) <<
						//1,          0,           0,
						//0, cos(rx), -sin(rx),
						//0, sin(rx),  cos(rx));

						//cv::Mat Ry = (cv::Mat_<double>(3,3) <<
						//cos(ry), 0, sin(ry),
						//0,       1,       0,
						//-sin(ry), 0, cos(ry));

						//cv::Mat Rz = (cv::Mat_<double>(3,3) <<
						//cos(rz), -sin(rz), 0,
						//sin(rz),  cos(rz), 0,
						//0,        0,       1);

						//R = Rz * Ry * Rx;

						cv::Mat R = get_R_from_tm12(actual_pose[3], actual_pose[4], actual_pose[5]);

						R.copyTo(robot_transform(cv::Rect(0,0,3,3)));
						robot_poses.push_back(robot_transform);

						// 將位姿資訊儲存到 YAML 檔案
						YAML::Emitter pose_out;
						pose_out << YAML::BeginMap;
						pose_out << YAML::Key << "image_" + std::to_string(valid_image_count);
						pose_out << YAML::Value;
						pose_out << YAML::BeginMap;
						pose_out << YAML::Key << "robot_pose" << YAML::Value << YAML::Flow << YAML::BeginSeq;
						for (const auto& val : actual_pose) {
							pose_out << val;
						}
						pose_out << YAML::EndSeq;

						// 儲存角點座標和對應深度
						pose_out << YAML::Key << "num_corners" << YAML::Value << corners.size();

						pose_out << YAML::Key << "corners(u,v), z" << YAML::Value << YAML::BeginSeq;
						for(size_t i = 0; i < corners.size(); i++) {
							pose_out << YAML::Flow << YAML::BeginSeq;
							pose_out << corners[i].x << corners[i].y << corner_depths[i];
							pose_out << YAML::EndSeq;
						}
						pose_out << YAML::EndSeq;

						pose_out << YAML::EndMap;
						pose_out << YAML::EndMap;

						std::ofstream fout((image_dir / (image_filename + "_pose.yaml")).string());
						fout << pose_out.c_str();
						fout.close();
					}
				}

				topHomingExecute();

				/*
				出廠預設：(解析度: 640x480)
				內參：
				616.0471801757812, 0.0, 322.584228515625
				0.0, 614.3076171875, 231.19287109375
				0.0, 0.0, 1.0
				畸變：
				0,0,0,0,0
				*/
				// 4. Calculate camera matrix
				cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) <<
				9.1558717337338635e+02, 0., 6.4301915357235964e+02,
				0., 9.1614100406855903e+02, 3.4930774356218546e+02,
				0., 0., 1.);
				cv::Mat distCoeffs = (cv::Mat_<double>(1,5) <<
				0.0, 0.0,
				0.0, 0.0,
				0.0);

				double error = cv::calibrateCamera(objectPoints, imagePoints, rgb_image_.size(),
				cameraMatrix, distCoeffs, rvecsCamera, tvecsCamera, cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 | cv::CALIB_ZERO_TANGENT_DIST,
				cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 100, DBL_EPSILON));
				RCLCPP_INFO(this->get_logger(), "Calibration error: %.3f", error);

				// 5. Hand-eye calibration
				std::vector<cv::Mat> camera_poses;
				std::vector<cv::Mat> camera_rotations;
				std::vector<cv::Mat> camera_translations;
				for(size_t i = 0; i < rvecsCamera.size(); i++) {
					cv::Mat R;
					cv::Rodrigues(rvecsCamera[i], R);
					cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
					R.copyTo(T(cv::Rect(0,0,3,3)));
					tvecsCamera[i].copyTo(T(cv::Rect(3,0,1,3)));
					camera_poses.push_back(T);

					// 儲存旋轉矩陣和平移向量
					camera_rotations.push_back(R);
					camera_translations.push_back(tvecsCamera[i]);
				}
				// 從機器人姿態中提取旋轉和平移
				std::vector<cv::Mat> robot_rotations;
				std::vector<cv::Mat> robot_translations;
				for(const auto& pose : robot_poses) {
					// 提取旋轉矩陣
					cv::Mat R = pose(cv::Rect(0,0,3,3));
					robot_rotations.push_back(R);

					// 提取平移向量
					cv::Mat t = pose(cv::Rect(3,0,1,3));
					robot_translations.push_back(t);
				}

				cv::Mat hand_eye_rotations, hand_eye_translations;
				cv::calibrateHandEye(
					robot_rotations,      // 機器人旋轉矩陣向量
					robot_translations,   // 機器人平移向量向量
					camera_rotations,     // 相機旋轉矩陣向量
					camera_translations,  // 相機平移向量向量
					hand_eye_rotations,   // 輸出的手眼校正轉換矩陣
					hand_eye_translations, // 輸出的手眼校正平移向量
					cv::CALIB_HAND_EYE_TSAI   // using Tsai's method
				); // ANDREFF not good

				// 6. Chessborad corners in base frame
				// 儲存所有角點在基底座標系下的位置
				std::vector<std::vector<cv::Point3d>> base_frame_corners;

				// 對每個影像進行處理
				for(size_t frame = 0; frame < imagePoints.size(); frame++) {
					std::vector<cv::Point3d> frame_corners_base;

					// 1. 反投影到相機座標系
					std::vector<cv::Point2d> image_points_double;
					for(size_t i = 0; i < imagePoints[frame].size(); i++) {
						image_points_double.push_back(cv::Point2d(imagePoints[frame][i].x, imagePoints[frame][i].y));
					}
					std::vector<cv::Point3d> camera_points;
					for(size_t i = 0; i < image_points_double.size(); i++) {
						// 使用深度資訊進行反投影
						double depth = imagePointsDepth[frame][i];
						if(depth <= 0) continue; // 跳過無效深度值

						// 使用針孔相機模型反投影
						double x = (image_points_double[i].x - cameraMatrix.at<double>(0,2)) / cameraMatrix.at<double>(0,0);
						double y = (image_points_double[i].y - cameraMatrix.at<double>(1,2)) / cameraMatrix.at<double>(1,1);

						// 相機座標系下的3D點
						cv::Point3d pt_camera(x * depth, y * depth, depth);
						camera_points.push_back(pt_camera);
					}

					// 2. 相機座標系到工具座標系的轉換
					cv::Mat R_cam_tool = hand_eye_rotations;
					cv::Mat t_cam_tool = hand_eye_translations;

					std::vector<cv::Point3d> tool_points;
					for(const auto& pt_camera : camera_points) {
						cv::Mat pt_cam = (cv::Mat_<double>(4,1) << pt_camera.x, pt_camera.y, pt_camera.z, 1);

						// 構建從相機到工具的變換矩陣
						cv::Mat T_cam_tool = cv::Mat::eye(4, 4, CV_64F);
						R_cam_tool.copyTo(T_cam_tool(cv::Rect(0,0,3,3)));
						t_cam_tool.copyTo(T_cam_tool(cv::Rect(3,0,1,3)));

						// 轉換到工具座標系
						cv::Mat pt_tool = T_cam_tool * pt_cam;

						tool_points.push_back(cv::Point3d(pt_tool.at<double>(0)/pt_tool.at<double>(3),
						pt_tool.at<double>(1)/pt_tool.at<double>(3),
						pt_tool.at<double>(2)/pt_tool.at<double>(3)));
					}

					// 3. 工具座標系到基底座標系的轉換
					cv::Mat T_tool_base = robot_poses[frame];

					for(const auto& pt_tool : tool_points) {
						bool ab = 1;
						cv::Mat pt_t = (cv::Mat_<double>(4,1) << pt_tool.x, pt_tool.y, pt_tool.z, 1);
						cv::Mat pt_base = T_tool_base * pt_t;

						frame_corners_base.push_back(cv::Point3d(pt_base.at<double>(0)/pt_base.at<double>(3),
						pt_base.at<double>(1)/pt_base.at<double>(3),
						pt_base.at<double>(2)/pt_base.at<double>(3)));
					}

					base_frame_corners.push_back(frame_corners_base);
				}

				// 7. Save calibration result
				res->ok = true;

				// 構建完整的檔案路徑
				std::filesystem::path result_file = save_dir / "camera_calibration_result.yaml";
				res->result_path = result_file.string();

				// 將結果寫入 YAML 文件
				cv::FileStorage fs(result_file.string(), cv::FileStorage::WRITE);
				fs << "error" << error;
				fs << "cameraMatrix" << cameraMatrix;
				fs << "distCoeffs" << distCoeffs;
				//fs << "newCameraMatrix" << newCameraMatrix;
				//fs << "mapx" << mapx;
				//fs << "mapy" << mapy;
				fs << "handEyeRotation" << hand_eye_rotations;
				fs << "handEyeTranslation" << hand_eye_translations;
				fs.release();

				// 將基底座標系下的角點位置寫入校正結果檔案
				YAML::Emitter corners_out;
				corners_out << YAML::BeginMap;
				corners_out << YAML::Key << "corners_in_base_frame(x,y,z)" << YAML::Value << YAML::BeginSeq;

				for(size_t frame = 0; frame < base_frame_corners.size(); frame++) {
					corners_out << YAML::BeginMap;
					corners_out << YAML::Key << "frame_" + std::to_string(frame + 1);
					corners_out << YAML::Value << YAML::BeginSeq;

					for(const auto& corner : base_frame_corners[frame]) {
						corners_out << YAML::Flow << YAML::BeginSeq;
						corners_out << YAML::Value << corner.x;
						corners_out << YAML::Value << corner.y;
						corners_out << YAML::Value << corner.z;
						corners_out << YAML::EndSeq;
					}

					corners_out << YAML::EndSeq;
					corners_out << YAML::EndMap;
				}

				corners_out << YAML::EndSeq;
				corners_out << YAML::EndMap;

				// 儲存檔案
				std::filesystem::path corners_file = save_dir / "corners_in_base_frame.yaml";
				std::ofstream corners_fout(corners_file.string());
				corners_fout << corners_out.c_str();
				corners_fout.close();

			}
		catch(const std::exception& e)
		{
			RCLCPP_ERROR(this->get_logger(), "Calibration failed: %s", e.what());
			res->ok = false;
			res->result_path = std::string("");
		}
	}

	void calibrationExecute_v2(const std::shared_ptr<Calibration::Request>& req, std::shared_ptr<Calibration::Response>& res) {
		RCLCPP_INFO(this->get_logger(), "Calibration V2");
		try
		{
			myCalibration calibrator("/workspaces/Hardware/src/ncku_csie_rl/tm12_amm/config/calibration_camera_eih_001.yaml");

			for(const auto& pose : calibrator.getTrajectory())
			{
				auto result = callTMDriverSetPosition(pose, 2,
					1.0, 1.0, 0, 1, 3).get();
					if(!result->ok) {
						RCLCPP_ERROR(this->get_logger(), "Failed to move robot to calibration pose");
						continue;
					}
					// 等待機器人到達位置和相機穩定
					//std::this_thread::sleep_for(std::chrono::seconds(2));

					std::array<double, 6> actual_pose;
					std::copy(tm12_feedback_state_->tool_pose.begin(),
					tm12_feedback_state_->tool_pose.end(),
					actual_pose.begin());

					// 儲存相機圖像
					cv::Mat rgb = rgb_image_.clone(),
					depth = depth_image_.clone();

					calibrator.append_data(rgb, depth, actual_pose);
					std::this_thread::sleep_for(std::chrono::seconds(2));
			}
			topHomingExecute();

			calibrator.execute();

			res->ok = true;
			res->result_path = calibrator.getResultPath();
		}
		catch(const std::exception& e)
		{
			RCLCPP_ERROR(this->get_logger(), "Calibration failed: %s", e.what());
			res->ok = false;
			res->result_path = std::string("");
		}
	}

	void handleVerifyCalibrationRequest(const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> res) {
		try {
			verifyCalibrationExecute(req, res);
		} catch (const std::exception& e) {
			RCLCPP_ERROR(this->get_logger(), "Exception in handleVerifyCalibrationRequest: %s", e.what());
		}
	}

	void verifyCalibrationExecute(const std::shared_ptr<Trigger::Request>& req, std::shared_ptr<Trigger::Response>& res) {
		RCLCPP_INFO(this->get_logger(), "Verify calibration");
		try
		{
			RCLCPP_INFO(this->get_logger(), "Verify calibration");

			// 1. 移動到拍照 pose
			auto result = callTMDriverSetPosition(pose_take_photo_).get();
			if(!result->ok) {
				RCLCPP_ERROR(this->get_logger(), "Failed to move robot to take photo pose");
				res->success = false;
				return;
			}

			// 2. 拍照
			std::this_thread::sleep_for(std::chrono::seconds(3));
			cv::Mat rgb = rgb_image_;
			cv::Mat depth = depth_image_;

			cv::Mat gray;
			cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

			std::array<double, 6> actual_pose;
			std::copy(tm12_feedback_state_->tool_pose.begin(),
			tm12_feedback_state_->tool_pose.end(),
			actual_pose.begin());

			// 5. 找角點
			std::vector<cv::Point2f> corners;
			bool found = cv::findChessboardCorners(gray, cv::Size(rgb.cols,rgb.rows), corners);
			std::vector<float> corner_depths;


			if(found) {
				// 在圖像上繪製角點
				cv::Mat corner_img = rgb.clone();
				cv::drawChessboardCorners(corner_img, cv::Size(rgb.cols,rgb.rows), corners, found);

				for(const auto& corner : corners) {
					// 從深度圖像中取得深度值(單位:米)
					float depth_value = depth.at<float>(corner.y, corner.x);

					corner_depths.push_back(static_cast<float>(depth_value));
				}

				// 精細化角點位置
				cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
				cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

			}

			// 5.1. 反投影到相機座標系
			std::vector<cv::Mat> T_obj2cam_set;
			cv::Mat R_target2cam = (cv::Mat_<double>(3,3) <<
			0., -1., 0.,
			-1., 0., 0.,
			0., 0., -1.);

			std::vector<cv::Point3d> camera_points;
			for(size_t i = 0; i < corners.size(); i++) {
				// 使用深度資訊進行反投影
				double depth = corner_depths[i];
				if(depth <= 0) continue; // 跳過無效深度值

				// 使用針孔相機模型反投影
				double x = (corners[i].x - new_camera_matrix_.at<double>(0,2)) / new_camera_matrix_.at<double>(0,0);
				double y = (corners[i].y - new_camera_matrix_.at<double>(1,2)) / new_camera_matrix_.at<double>(1,1);

				// 相機座標系下的3D點
				cv::Point3d pt_camera(x * depth, y * depth, depth);
				camera_points.push_back(pt_camera);

				cv::Mat T_target2cam = cv::Mat::eye(4, 4, CV_64F);
				R_target2cam.copyTo(T_target2cam(cv::Rect(0,0,3,3)));
				cv::Mat t_target2cam = (cv::Mat_<double>(3,1) << x * depth, y * depth, depth);
				t_target2cam.copyTo(T_target2cam(cv::Rect(3,0,1,3)));

				T_obj2cam_set.push_back(T_target2cam);
			}

			// 使用實際回饋的機器人位姿
			cv::Mat T_g2b = cv::Mat::eye(4, 4, CV_64F);
			// 填充旋轉和平移
			for(int i = 0; i < 3; i++) {
				T_g2b.at<double>(i,3) = actual_pose[i];
			}

			cv::Mat R = get_R_from_tm12(actual_pose[3], actual_pose[4], actual_pose[5]);

			R.copyTo(T_g2b(cv::Rect(0,0,3,3)));



			// 6. 將角點轉到基底座標系

			cv::Mat T_cam2gripper = cv::Mat::eye(4, 4, CV_64F);
			// 填充旋轉和平移
			for(int i = 0; i < 3; i++) {
				T_cam2gripper.at<double>(i,3) = t_c2g_.at<double>(i,0);
			}
			R_c2g_.copyTo(T_cam2gripper(cv::Rect(0,0,3,3)));



			std::vector<cv::Mat> T_obj2base_set;
			for(const auto& T_obj2cam : T_obj2cam_set) {
				cv::Mat T_obj2base = T_g2b * T_cam2gripper * T_obj2cam;
				T_obj2base_set.push_back(T_obj2base);
			}

			// 7. 轉成TM 表達
			std::vector<std::array<double, 6>> tm_poses;
			for(const auto& T : T_obj2base_set) {
				std::array<double, 6> tm_pose;

				// 提取平移向量
				for(int i = 0; i < 3; i++) {
					tm_pose[i] = T.at<double>(i,3);
				}

				// 提取旋轉矩陣並轉換為歐拉角
				cv::Mat R = T(cv::Rect(0,0,3,3));
				auto euler = get_euler_from_R(R);

				// 存入歐拉角
				tm_pose[3] = -M_PI;//euler[0]; // rx
				tm_pose[4] = 0;//euler[1]; // ry
				tm_pose[5] = M_PI_4;//euler[2]; // rz

				// 7. 調整筆長
				tm_pose[2]+=10;
				RCLCPP_INFO(this->get_logger(), "tm_pose: %f, %f, %f, %f, %f, %f", tm_pose[0], tm_pose[1], tm_pose[2], tm_pose[3], tm_pose[4], tm_pose[5]);
				tm_poses.push_back(tm_pose);
			}

			// 8. 移動機器人
			for(const auto& pose : tm_poses) {
				auto result = callTMDriverSetPosition(pose, 2,
					1.0, 1.0, 100, 1, 1).get();
					if(!result->ok) {
						RCLCPP_ERROR(this->get_logger(), "Failed to move robot to calibration pose");
						continue;
					}
					// 等待機器人到達位置和相機穩定
					std::this_thread::sleep_for(std::chrono::seconds(3));
			}

		}catch(const std::exception& e){
			RCLCPP_ERROR(this->get_logger(), "Verify calibration failed: %s", e.what());
			res->success = false;
		}
		res->success = true;
	}


	// 將旋轉矩陣轉換為 TM robot 的歐拉角表示 (EulerZYX, radians)
	std::array<double, 3> get_euler_from_R(const cv::Mat& R) {
		std::array<double, 3> euler = {0.0, 0.0, 0.0}; // [rx, ry, rz]

		// 檢查矩陣維度
		if(R.rows != 3 || R.cols != 3) {
			throw std::runtime_error("Rotation matrix must be 3x3");
		}

		// 從旋轉矩陣提取歐拉角 (ZYX順序)
		// rz (yaw)
		euler[2] = std::atan2(R.at<double>(1,0), R.at<double>(0,0));

		// ry (pitch)
		double sy = -R.at<double>(2,0);
		euler[1] = std::asin(sy);

		// rx (roll)
		euler[0] = std::atan2(R.at<double>(2,1), R.at<double>(2,2));

		return euler;
	}

////////////////////////////////////////////////////////////////////////////////////////////////////////


};

int main(int argc, char** argv) {
	try {
		rclcpp::init(argc, argv);
		auto node = std::make_shared<TM12_AMM_old>();

		rclcpp::executors::MultiThreadedExecutor executor;
		executor.add_node(node);

		while (rclcpp::ok()) {
			executor.spin();
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}

		rclcpp::shutdown();
	} catch (const std::exception& e) {
		RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
		return 1;
	} catch (...) {
		RCLCPP_ERROR(rclcpp::get_logger("main"), "Unknown exception occurred");
		return 1;
	}
	return 0;
}

