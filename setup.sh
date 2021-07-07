#!/bin/bash

sudo apt-get update
sudo apt-get install -y ros-kinetic-dbw-mkz-msgs
cd /home/workspace/CarND-Capstone/ros
rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y

cd /home/workspace/CarND-Capstone
pip install -r requirements.txt

pip uninstall -y catkin_pkg
pip install -U catkin_pkg==0.4.10

cd ros
find $PWD -type f -iname "*.py" -exec chmod +x {} \;
catkin_make clean
catkin_make
