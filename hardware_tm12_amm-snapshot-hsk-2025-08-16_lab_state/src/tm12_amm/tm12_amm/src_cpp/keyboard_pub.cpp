#include <stdio.h>
#include <unistd.h>
#include <termios.h>
#include <sys/select.h>

#include <map>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/char.hpp"

// Reminder message
const char* msg = R"(
Reading from the keyboard and Publishing to Char!
---------------------------
ESC: homing

q/a: +x/-x      r/f: +rx/-rx
w/s: +y/-y      t/g: +ry/-ry
e/d: +z/-z      y/h: +rz/-rz

z/x: closing/opening gripper
---------------------------
Press any key to send it as a message.
CTRL+C to quit.
)";

// For non-blocking keyboard inputs
int getch(void)
{
  int ch = -1;
  struct termios oldt;
  struct termios newt;
  fd_set set;
  struct timeval timeout;

  // Store old settings, and copy to new settings
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;

  // Make required changes and apply the settings
  newt.c_lflag &= ~(ICANON | ECHO);
  newt.c_iflag |= IGNBRK;
  newt.c_iflag &= ~(INLCR | ICRNL | IXON | IXOFF);
  newt.c_lflag &= ~(ICANON | ECHO | ECHOK | ECHOE | ECHONL | ISIG | IEXTEN);
  newt.c_cc[VMIN] = 1;
  newt.c_cc[VTIME] = 0;
  tcsetattr(fileno(stdin), TCSANOW, &newt);

  // Initialize the file descriptor set
  FD_ZERO(&set);
  FD_SET(STDIN_FILENO, &set);

  // Initialize the timeout data structure
  timeout.tv_sec = 0;
  timeout.tv_usec = 100; // 100 milliseconds

  // Check if there is input available
  int rv = select(STDIN_FILENO + 1, &set, NULL, NULL, &timeout);
  if (rv > 0) {
    // Get the current character
    ch = getchar();
  }

  // Reapply old settings
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  return ch;
}

int main(int argc, char** argv){

  // node init
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("keyboard_input");
  // define publisher
  auto _pub = node->create_publisher<std_msgs::msg::Char>("/keyboard_manual", 10);

  std_msgs::msg::Char char_msg;
  printf("%s", msg);

  while (rclcpp::ok()) {
    // get the pressed key
    int key = getch();

    // If a key was pressed
    if (key != -1) {
      // If ctrl-C (^C) was pressed, terminate the program
      if (key == '\x03') {
        printf("\nExiting...\n");
        break;
      }

      // Set the character message
      char_msg.data = key;

      // Publish the character message
      _pub->publish(char_msg);
      printf("\rPublished: %c   ", key);
    }

    rclcpp::spin_some(node);
  }

  return 0;
}