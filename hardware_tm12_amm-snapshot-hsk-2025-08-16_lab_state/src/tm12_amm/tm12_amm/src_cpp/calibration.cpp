#include "tm12_amm/calibration.hpp"


myCalibration::myCalibration(const std::string& config_path)
: config_path_(config_path)
{
    // 讀取配置檔案
    if (!readConfig()) {
            throw std::runtime_error("Failed to read calibration config");
    }

    initializeObjectCorners();
}