#!/usr/bin/env python3
"""Apply three fixes to the Weed-Zap package (run from the package folder):

    python3 tools/apply_fixes.py

1. camera_droid/area_patrol.py  - robot no longer stops for good:
     * leaving the geofence now drives it back inside instead of finishing
     * after the last lane it starts another lap (parameter `loop`, default true)
     * stuck detection: backs up and re-aims if it hasn't moved for `stuck_time` s
2. setup.py                     - installs the whole models/ tree
3. launch/sim.launch.py         - adds models/ to GZ_SIM_RESOURCE_PATH so model://... works

Each file is backed up as <file>.bak first. Re-running is safe (already-patched files are skipped).
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(rel, marker, edits):
    path = ROOT / rel
    text = path.read_text()
    if marker in text:
        print(f'[skip] {rel}: already patched')
        return
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f'[FAIL] {rel}: could not find a unique anchor:\n{old!r}\n'
                     'The file differs from the one I saw. Nothing was changed in it.')
        text = text.replace(old, new)
    shutil.copy(path, str(path) + '.bak')
    path.write_text(text)
    print(f'[ok]   {rel}')


# ---------------------------------------------------------------- area_patrol
patrol_edits = [
    # new parameters
    ("        self.declare_parameter('crop_hit_radius', 0.17)  # blade reach (0.12) + crop radius (0.05) (m)\n",
     "        self.declare_parameter('crop_hit_radius', 0.17)  # blade reach (0.12) + crop radius (0.05) (m)\n"
     "        self.declare_parameter('loop', True)             # keep patrolling lap after lap\n"
     "        self.declare_parameter('stuck_time', 6.0)        # s without moving before backing up\n"),
    ("        self.crop_hit_radius = gp('crop_hit_radius').value\n",
     "        self.crop_hit_radius = gp('crop_hit_radius').value\n"
     "        self.loop = gp('loop').value\n"
     "        self.stuck_time = gp('stuck_time').value\n"),
    # state
    ("        self.finished = False\n\n        self.cmd_pub",
     "        self.finished = False\n"
     "        self.stuck_ref = None        # (x, y, t) used by stuck detection\n"
     "        self.escape_until = 0.0\n\n        self.cmd_pub"),
    # geofence: recover instead of stopping forever
    ("""        if not (self.x_min - m <= x <= self.x_max + m and
                self.y_min - m <= y <= self.y_max + m):
            self.stop()
            self.spin_blade(0.0)
            self.get_logger().error('Outside geofence! Stopping.')
            self.finished = True
            return
""",
     """        if not (self.x_min - m <= x <= self.x_max + m and
                self.y_min - m <= y <= self.y_max + m):
            self.spin_blade(0.0)
            self.get_logger().warn('Outside geofence - driving back inside',
                                   throttle_duration_sec=2.0)
            self.mode = 'PATROL'
            self.state = 'TURN'
            cx, cy = (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2
            err = wrap(math.atan2(cy - y, cx - x) - yaw)
            cmd = Twist()
            if abs(err) > 0.3:
                cmd.angular.z = clip(1.5 * err, -self.ang, self.ang)
            else:
                cmd.linear.x = 0.2
                cmd.angular.z = clip(1.5 * err, -self.ang, self.ang)
            self.cmd_pub.publish(cmd)
            return
"""),
    # next lap instead of finishing + back-up manoeuvre
    ("            self.finished = True\n            return\n\n        tx, ty = self.waypoints[self.idx]\n",
     """            if self.loop:
                self.get_logger().info('Starting another lap')
                self.waypoints.reverse()        # come back along the same lanes
                self.idx = 0
                self.state = 'TURN'
                return
            self.finished = True
            return

        if self.now() < self.escape_until:        # backing away from an obstacle
            esc = Twist()
            esc.linear.x = -0.2
            esc.angular.z = 0.6
            self.cmd_pub.publish(esc)
            return

        tx, ty = self.waypoints[self.idx]
"""),
    ("        if self.state == 'TURN':\n            if abs(err) < 0.05:\n",
     "        if self.state == 'TURN':\n            self.stuck_ref = None\n            if abs(err) < 0.05:\n"),
    ("                cmd.linear.x = min(self.lin, 0.8 * dist + 0.05)\n",
     """                if (self.stuck_ref is None or
                        math.hypot(x - self.stuck_ref[0], y - self.stuck_ref[1]) > 0.05):
                    self.stuck_ref = (x, y, self.now())
                elif self.now() - self.stuck_ref[2] > self.stuck_time:
                    self.get_logger().warn('Robot seems stuck - backing up and re-aiming')
                    self.escape_until = self.now() + 1.5
                    self.stuck_ref = None
                    self.state = 'TURN'
                cmd.linear.x = min(self.lin, 0.8 * dist + 0.05)
"""),
]

# ------------------------------------------------------------------- setup.py
setup_edits = [
    ("package_name = 'camera_droid'\n",
     """package_name = 'camera_droid'


def tree(src):
    \"\"\"Install every file under `src` (keeps the folder structure).\"\"\"
    out = []
    for root, _, files in os.walk(src):
        if files:
            out.append((os.path.join('share', package_name, root),
                        [os.path.join(root, f) for f in files]))
    return out
"""),
    ("    ],\n    install_requires=['setuptools'],",
     "    ] + tree('models'),\n    install_requires=['setuptools'],"),
]

# ---------------------------------------------------------------- launch file
launch_edits = [
    ("from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription",
     "from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,\n"
     "                            IncludeLaunchDescription)"),
    ("    return LaunchDescription(args + [gz_sim, rsp, spawn, bridge, remove_bridge, detector, patrol])",
     "    # lets world files use <uri>model://soil_ground</uri> etc. from this package's models/\n"
     "    models_env = AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(pkg, 'models'))\n\n"
     "    return LaunchDescription([models_env] + args +\n"
     "                             [gz_sim, rsp, spawn, bridge, remove_bridge, detector, patrol])"),
]

if __name__ == '__main__':
    patch('camera_droid/area_patrol.py', "self.stuck_ref = None        # (x, y, t)", patrol_edits)
    patch('setup.py', "def tree(src):", setup_edits)
    patch('launch/sim.launch.py', "AppendEnvironmentVariable", launch_edits)
    print('\nNow rebuild:  colcon build  &&  source install/setup.bash')
