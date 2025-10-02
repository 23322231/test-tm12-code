cmake_minimum_required(VERSION 3.12)
project(tm12_amm)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(cv_bridge REQUIRED)
find_package(Curses REQUIRED)

find_package(grpr2f85_ifaces REQUIRED)
find_package(tm_msgs REQUIRED)
find_package(realsense2_camera_msgs REQUIRED)
find_package(tm12_amm_interfaces)

#find_package(rqt_gui REQUIRED)
#find_package(rqt_gui_cpp REQUIRED)
#find_package(pluginlib REQUIRED)
#find_package(qt_gui_cpp REQUIRED)

# Include Cpp "include" directory
include_directories(include)

# Create Cpp executable
add_executable(cpp_hello_world src_cpp/cpp_node.cpp)

add_executable(tm12_amm_cpp src_cpp/tm12_amm.cpp)

add_executable(keyboard_pub_cpp src_cpp/keyboard_pub.cpp)

target_link_libraries(tm12_amm_cpp
                      Eigen3::Eigen
                      ${OpenCV_LIBS}
                      ${CURSES_LIBRARIES}
                      yaml-cpp
                      )

ament_target_dependencies(cpp_hello_world rclcpp)

ament_target_dependencies(tm12_amm_cpp
                          Eigen3
                          OpenCV
                          rclcpp
                          rclcpp_action
                          rclcpp_components
                          action_msgs
                          std_msgs
                          sensor_msgs
                          control_msgs
                          std_srvs
                          cv_bridge
                          grpr2f85_ifaces
                          tm_msgs
                          realsense2_camera_msgs
                          tm12_amm_interfaces
                          )

ament_target_dependencies(keyboard_pub_cpp
                          Eigen3
                          OpenCV
                          rclcpp
                          rclcpp_action
                          rclcpp_components
                          action_msgs
                          std_msgs
                          sensor_msgs
                          control_msgs
                          std_srvs
                          cv_bridge
                          grpr2f85_ifaces
                          tm_msgs
                          realsense2_camera_msgs
                          tm12_amm_interfaces
                          )

# Install Cpp executables
install(TARGETS
  cpp_hello_world
  tm12_amm_cpp
  keyboard_pub_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
#ament_python_install_package()

# Install Python executables
install(PROGRAMS
  src_py/tm12_amm.py
  src_py/calibration.py
  src_py/keyboard_pub.py
  DESTINATION lib/${PROJECT_NAME}
)

install(# launch
  DIRECTORY launch config xacro
  DESTINATION share/${PROJECT_NAME}
  )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
