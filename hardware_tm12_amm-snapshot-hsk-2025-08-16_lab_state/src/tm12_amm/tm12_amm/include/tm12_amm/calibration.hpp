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

			// 設置時間戳記
			auto now = std::chrono::system_clock::now();
			auto now_time_t = std::chrono::system_clock::to_time_t(now);
			std::stringstream ss;
			ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
			timestamp_ = ss.str();

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
	std::vector<cv::Mat> robot_poses_;

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
            std::cerr << "Failed to read config: " << e.what() << std::endl;
			return false;
		}
	}

	// 建立目錄結構
	bool createDirectories() {
		try {
			std::filesystem::path save_dir("/workspaces/AI_Robot_ws/Hardware/src/ncku_csie_rl/tm12_amm/data/calibration");
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

			double error = cv::calibrateCamera(objectPoints_, imagePoints_, imageSize,
				result_.cameraMatrix, result_.distCoeffs, rvecsCamera_, tvecsCamera_);
				std::cout << "Camera calibration error: " << error << std::endl;

				// 計算最佳的新相機矩陣
				//newCameraMatrix_ = cv::getOptimalNewCameraMatrix(cameraMatrix_, distCoeffs_,
				//                                               imageSize, 0, imageSize);

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

			// 準備相機姿態數據
			for(size_t i = 0; i < rvecsCamera_.size(); i++) {
				cv::Mat R;
				cv::Rodrigues(rvecsCamera_[i], R);
				camera_rotations.push_back(R);
				camera_translations.push_back(tvecsCamera_[i]);
			}

			// 準備機器人姿態數據
			for(const auto& pose : robotPoses_) {
				cv::Mat R = getRotationFromTm12(pose[3], pose[4], pose[5]);
				cv::Mat t = (cv::Mat_<double>(3,1) << pose[0], pose[1], pose[2]);

				robot_rotations.push_back(R);
				robot_translations.push_back(t);
			}

			// 執行手眼校正
			cv::calibrateHandEye(
				robot_rotations,
				robot_translations,
				camera_rotations,
				camera_translations,
				result_.handEyeRotations,
				result_.handEyeTranslations,
				cv::CALIB_HAND_EYE_TSAI
			);

			return true;
		}
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
