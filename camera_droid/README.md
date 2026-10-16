# camera_droid – weed-spotting, weed-cutting droid (ROS 2 Jazzy + Gazebo Harmonic)

    ros2 launch camera_droid sim.launch.py
    ros2 launch camera_droid sim.launch.py x_min:=-3.0 x_max:=3.0 y_min:=-2.0 y_max:=2.0
    ros2 launch camera_droid sim.launch.py cut_weeds:=false      # patrol + photos only
    ros2 launch camera_droid sim.launch.py patrol:=false         # drive by hand (teleop_twist_keyboard)

Output: ~/droid_photos/run_<time>/  (lane photos, weeds/ before+after photos, weeds_report.csv)
Watch: ros2 run rqt_image_view rqt_image_view /weeds/annotated
Manual blade test: ros2 topic pub --once /blade/spin std_msgs/msg/Float64 "{data: 40.0}"
