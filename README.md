# Weed-Zap — Quick Start (Ubuntu Terminal)

**All commands below are run in the Ubuntu Terminal**, not in the Windows Command Prompt or PowerShell.

Assumes you already installed **Ubuntu 24.04, ROS 2 Jazzy, and Gazebo Harmonic**.

## 1. Open the Ubuntu Terminal and extract the project

If the ZIP is in your Ubuntu `Downloads` folder:

```bash
mkdir -p ~/weed_zap_ws/src
cd ~/weed_zap_ws/src
unzip ~/Downloads/camera_droid_with_crops.zip
```

The extracted folder should be `camera_droid`.

## 2. Install project dependencies

Run these commands in the Ubuntu Terminal:

```bash
sudo apt update
sudo apt install -y ros-jazzy-ros-gz ros-jazzy-robot-state-publisher \
  ros-jazzy-rqt-image-view python3-opencv python3-numpy python3-yaml \
  python3-colcon-common-extensions
```

## 3. Build the project

```bash
source /opt/ros/jazzy/setup.bash
cd ~/weed_zap_ws
colcon build --symlink-install
source install/setup.bash
```

## 4. Run the simulation

```bash
ros2 launch camera_droid sim.launch.py
```

Gazebo should open with the rover and plant arena. Keep this terminal open while it runs.

Optional commands:

```bash
ros2 launch camera_droid sim.launch.py cut_weeds:=false
ros2 launch camera_droid sim.launch.py patrol:=false
```

## 5. View the camera feed

Open a **second Ubuntu Terminal window**, then run:

```bash
source /opt/ros/jazzy/setup.bash
source ~/weed_zap_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view /weeds/annotated
```

## 6. Find the results

Photos and CSV reports are saved under:


```bash
~/droid_photos/
```

## Troubleshooting

- **`unzip: command not found`:** run `sudo apt install unzip`.
- **Package not found:** source ROS 2, rebuild in `~/weed_zap_ws`, then source `install/setup.bash`.
- **Gazebo or ROS errors:** check that you are using compatible ROS 2 Jazzy and Gazebo Harmonic installations.

## Important notes

- The detector currently uses colour thresholds and blob size; it is not a trained AI model.
- Weed removal is simulated in Gazebo, not performed by real hardware.
- Test the project in your Ubuntu environment before claiming that it works.
