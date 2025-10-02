#!/bin/bash

xhost +local:docker

if ls /dev/ttyUSB* 1> /dev/null 2>&1; then
    echo "ttyUSB device found"
    sudo chmod 666 /dev/ttyUSB*
else
    echo "No ttyUSB device found."
fi

if ls /dev/video* 1> /dev/null 2>&1; then
    echo "video device found"
    sudo chmod 666 /dev/video*
else
    echo "No video device found."
fi
