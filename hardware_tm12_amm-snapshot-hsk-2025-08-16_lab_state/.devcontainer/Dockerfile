# Base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base
#FROM ubuntu:22.04 AS base

# Set shell and environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    QT_X11_NO_MITSHM=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    AMENT_PREFIX_PATH=/opt/ros/humble \
    COLCON_PREFIX_PATH=/opt/ros/humble \
    PATH=/opt/ros/humble/bin:$PATH \
    PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages \
    LD_LIBRARY_PATH=/opt/ros/humble/lib \
    ROS_DISTRO=humble \
    ROS_PACKAGE=desktop-full \
    EGL_PLATFORM=surfaceless

# Install system packages, Python packages, and librealsense
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic utilities
    locales locales-all tzdata sudo \
    build-essential software-properties-common gdb cmake git curl wget vim \
    manpages-dev gnupg2 lsb-release autoconf automake libtool bsdmainutils \
    dirmngr gosu iputils-ping less libssl-dev pkg-config \
    ca-certificates nano sed unzip tree zip openssh-client \
    # Python related
    python3 python3-py python3-serial python-is-python3 python3-pip \
    python3-numpy python3-pytest python3-setuptools python3-dev python3-venv \
    python3-argcomplete \
    # Libraries
    libprotobuf-dev protobuf-compiler zlib1g-dev swig valgrind \
    openssl libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 \
    libgl1-mesa-dev libjpeg-dev libpng-dev language-pack-en libgomp1 \
    libboost-all-dev libasio-dev libvulkan1 mesa-utils \
    # X11 related for GUI
    x11-apps libxcb-cursor0 xauth x11-utils xserver-xorg-core \
    libgl1-mesa-glx libgl1-mesa-dri libxcb1 libxcb-glx0 libxcb-dri2-0 \
    libxcb-dri3-0 libxcb-present0 libxcb-sync1 libxshmfence1 libxxf86vm1 \
    && \
    # Install librealsense
    mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    librealsense2-utils librealsense2-dev && \
    # Locale and timezone setup
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    # Python package manager and packages
    pip install --upgrade pip && \
    python3 -m pip install uv && \
    python3 -m uv pip install numpy==1.26.4 scipy pyserial pyads matplotlib && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install ROS
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep python3-vcstool python3-colcon-common-extensions \
    ros-${ROS_DISTRO}-${ROS_PACKAGE} ros-dev-tools \
    ros-${ROS_DISTRO}-joint-state-publisher ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-moveit ros-${ROS_DISTRO}-usb-cam \
    ros-${ROS_DISTRO}-librealsense2* ros-${ROS_DISTRO}-realsense2-* \
    ros-${ROS_DISTRO}-rtabmap-ros && \
    rosdep init && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# User setup
ENV USERNAME=ncku_csie_rl \
    USER_UID=1000 \
    USER_GID=1000
    #USERNAME=tienyu #

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get update && apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up bashrc for user (ROS sourcing and colcon autocompletion)
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc && \
    echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> ~/.bashrc && \
    echo "if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ]; then source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash; fi" >> ~/.bashrc

ENV AMENT_CPPCHECK_ALLOW_SLOW_VERSIONS=1
ENV QT_QPA_PLATFORM=xcb \
    NO_AT_BRIDGE=1 \
    LIBGL_ALWAYS_INDIRECT=0 \
    MESA_GL_VERSION_OVERRIDE=3.3 \
    LIBGL_ALWAYS_SOFTWARE=0

# Workspace setup
ENV WORKSPACE=/home/$USERNAME
WORKDIR $WORKSPACE

CMD ["bash"]