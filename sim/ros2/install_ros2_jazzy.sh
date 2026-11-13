#!/usr/bin/env bash
# Install ROS 2 Jazzy (Ubuntu 24.04 "noble") desktop + RViz inside WSL.
# Run from WSL:  bash install_ros2_jazzy.sh      (asks for your sudo password)
set -euo pipefail

sudo apt-get update
sudo apt-get install -y software-properties-common curl python3.12-venv
sudo add-apt-repository -y universe

# Official ROS 2 apt source (ros2-apt-source package, per docs.ros.org Jazzy install guide)
ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F '"tag_name"' | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt-get update
sudo apt-get install -y ros-jazzy-desktop ros-dev-tools

grep -q "/opt/ros/jazzy/setup.bash" ~/.bashrc || echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "Done. Open a new WSL shell (or 'source /opt/ros/jazzy/setup.bash') and run: ros2 doctor --report | head"
