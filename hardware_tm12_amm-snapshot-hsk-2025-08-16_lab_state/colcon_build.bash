colcon build  --merge-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -Wall -Wextra -Wpedantic --parallel-workers=4
sudo chmod 666 /dev/ttyUSB*
source install/setup.bash 
