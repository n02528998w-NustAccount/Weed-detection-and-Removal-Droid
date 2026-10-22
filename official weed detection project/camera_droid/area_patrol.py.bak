#!/usr/bin/env python3
"""Patrol a rectangular area, photograph weeds and cut them with the front blade.
Crops are recognised too: they are logged and photographed but never cut.

Coordinates are in the odom frame. The droid spawns at the world origin, so
odom == world coordinates in the provided arena.

Modes
  PATROL    drive lawn-mower lanes (photos at waypoints / every N metres)
  SETTLE    weed spotted: stop and wait for a fresh, steady detection
  ALIGN     turn to face the weed (+ "found" photos)
  APPROACH  creep until the weed is under the blade (blade_offset ahead of base_link)
  CUT       spin the blade; half-way through, the weed is removed from the sim
  POST      "after" photos, then back to PATROL
"""
import csv
import math
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

try:
    from ros_gz_interfaces.msg import Entity
    from ros_gz_interfaces.srv import DeleteEntity
except ImportError:          # fall back to the `gz service` command line
    Entity = DeleteEntity = None


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def clip(v, lo, hi):
    return max(lo, min(hi, v))


class AreaPatrol(Node):
    def __init__(self):
        super().__init__('area_patrol')

        # Area to cover (metres) and driving behaviour
        self.declare_parameter('x_min', -4.0)
        self.declare_parameter('x_max', 4.0)
        self.declare_parameter('y_min', -3.0)
        self.declare_parameter('y_max', 3.0)
        self.declare_parameter('lane_spacing', 2.0)
        self.declare_parameter('linear_speed', 0.4)
        self.declare_parameter('angular_speed', 0.8)
        self.declare_parameter('photo_distance', 1.5)
        self.declare_parameter('photo_dir', '~/droid_photos')
        self.declare_parameter('geofence_margin', 0.5)
        # Weed cutting
        self.declare_parameter('cut_weeds', True)
        self.declare_parameter('blade_offset', 0.45)     # blade centre ahead of base_link (m)
        self.declare_parameter('blade_speed', 40.0)      # rad/s
        self.declare_parameter('cut_duration', 2.0)      # s
        self.declare_parameter('settle_time', 0.8)       # s
        self.declare_parameter('weeds_file', '')
        self.declare_parameter('world_name', 'arena')
        self.declare_parameter('match_radius', 0.40)     # detection -> sim model match (m)
        self.declare_parameter('handled_radius', 0.35)   # ignore detections near cut weeds (m)
        # Crop protection
        self.declare_parameter('crops_file', '')         # ground truth, used only for the score
        self.declare_parameter('crop_clearance', 0.20)   # never cut a weed closer than this to a crop (m)
        self.declare_parameter('crop_merge_radius', 0.20)  # detections closer than this = same crop (m)
        self.declare_parameter('crop_hit_radius', 0.17)  # blade reach (0.12) + crop radius (0.05) (m)

        gp = self.get_parameter
        self.x_min, self.x_max = gp('x_min').value, gp('x_max').value
        self.y_min, self.y_max = gp('y_min').value, gp('y_max').value
        self.lin = gp('linear_speed').value
        self.ang = gp('angular_speed').value
        self.photo_dist = gp('photo_distance').value
        self.margin = gp('geofence_margin').value
        self.cut_weeds = gp('cut_weeds').value
        self.blade_offset = gp('blade_offset').value
        self.blade_speed = gp('blade_speed').value
        self.cut_duration = gp('cut_duration').value
        self.settle_time = gp('settle_time').value
        self.world = gp('world_name').value
        self.match_radius = gp('match_radius').value
        self.handled_radius = gp('handled_radius').value
        self.crop_clearance = gp('crop_clearance').value
        self.crop_merge = gp('crop_merge_radius').value
        self.crop_hit_radius = gp('crop_hit_radius').value

        run = datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self.out_dir = Path(gp('photo_dir').value).expanduser() / run
        self.weed_dir = self.out_dir / 'weeds'
        self.weed_dir.mkdir(parents=True, exist_ok=True)
        self.crop_dir = self.out_dir / 'crops'
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.crop_report_path = self.out_dir / 'crops_report.csv'
        with open(self.crop_report_path, 'w', newline='') as f:
            csv.writer(f).writerow(['crop_id', 'est_x', 'est_y', 'time_s', 'action'])
        self.report_path = self.out_dir / 'weeds_report.csv'
        with open(self.report_path, 'w', newline='') as f:
            csv.writer(f).writerow(['weed_id', 'sim_model', 'est_x', 'est_y', 'time_s'])

        self.weeds = self.load_weeds(gp('weeds_file').value)
        self.crops_truth = self.load_points(gp('crops_file').value)
        self.crops_seen = []         # [x, y, n_observations] in odom frame, de-duplicated
        self.crop_hits = 0           # cuts that put a crop inside the blade zone (should stay 0)
        self.crops_skipped = 0       # weed sightings ignored because a crop was too close
        self.warned_near_crop = set()
        self.waypoints = self.make_lanes(gp('lane_spacing').value)
        self.idx = 0
        self.state = 'TURN'          # sub-state while patrolling: TURN / DRIVE
        self.mode = 'PATROL'
        self.mode_t0 = 0.0
        self.pose = None             # (x, y, yaw)
        self.last_photo_xy = None
        self.front_img = None
        self.ground_img = None
        self.annot_img = None
        self.dets = []               # latest detections, base_link frame
        self.det_rx = -1.0
        self.target = None           # weed being cut (odom frame)
        self.removed = False
        self.handled = []
        self.photo_count = 0
        self.weed_id = 0
        self.cut_count = 0
        self.finished = False

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.blade_pub = self.create_publisher(Float64, 'blade/spin', 10)
        self.create_subscription(Odometry, 'odom', self.on_odom, 10)
        self.create_subscription(Image, 'camera/image_raw', self.on_front, qos_profile_sensor_data)
        self.create_subscription(Image, 'ground_camera/image_raw', self.on_ground,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, 'weeds/annotated', self.on_annot, qos_profile_sensor_data)
        self.create_subscription(PoseArray, 'weeds/detections', self.on_dets, 10)
        self.create_subscription(PoseArray, 'crops/detections', self.on_crops, 10)
        self.del_cli = None
        if DeleteEntity is not None:
            self.del_cli = self.create_client(DeleteEntity, f'/world/{self.world}/remove')
        self.create_timer(0.05, self.step)

        self.get_logger().info(
            f'Patrolling x[{self.x_min}, {self.x_max}] y[{self.y_min}, {self.y_max}] '
            f'with {len(self.waypoints)} waypoints, weed cutting '
            f'{"ON" if self.cut_weeds else "OFF"}. Output -> {self.out_dir}')

    # ---- setup helpers --------------------------------------------------
    def load_weeds(self, path):
        if not path:
            return []
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or []
            return [dict(name=w['name'], x=float(w['x']), y=float(w['y']), removed=False)
                    for w in data]
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'Could not read weeds file {path}: {e}')
            return []

    def load_points(self, path):
        if not path:
            return []
        try:
            with open(path) as f:
                return [(float(c['x']), float(c['y'])) for c in (yaml.safe_load(f) or [])]
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'Could not read {path}: {e}')
            return []

    def make_lanes(self, spacing):
        n = max(1, math.ceil((self.y_max - self.y_min) / spacing - 1e-6)) + 1
        pts = []
        for i in range(n):
            y = min(self.y_min + i * spacing, self.y_max)
            xs = (self.x_min, self.x_max) if i % 2 == 0 else (self.x_max, self.x_min)
            pts += [(xs[0], y), (xs[1], y)]
        return pts

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- callbacks ------------------------------------------------------
    def on_odom(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def on_front(self, msg):
        self.front_img = msg

    def on_ground(self, msg):
        self.ground_img = msg

    def on_annot(self, msg):
        self.annot_img = msg

    def on_dets(self, msg):
        self.dets = [(p.position.x, p.position.y) for p in msg.poses]
        self.det_rx = self.now()

    def on_crops(self, msg):
        """Crops are only ever recorded and photographed, never cut."""
        if self.pose is None:
            return
        for p in msg.poses:
            wx, wy = self.to_world(p.position.x, p.position.y)
            for c in self.crops_seen:
                if math.hypot(wx - c[0], wy - c[1]) < self.crop_merge:
                    c[2] += 1
                    c[0] += (wx - c[0]) / c[2]          # running mean of the position
                    c[1] += (wy - c[1]) / c[2]
                    break
            else:
                self.crops_seen.append([wx, wy, 1])
                n = len(self.crops_seen)
                self.get_logger().info(
                    f'Crop {n} spotted at ({wx:.2f}, {wy:.2f}) - leaving it alone')
                self.save_img(self.annot_img, self.crop_dir / f'crop{n:02d}_left_alone.png')
                with open(self.crop_report_path, 'a', newline='') as f:
                    csv.writer(f).writerow([n, f'{wx:.3f}', f'{wy:.3f}', f'{self.now():.1f}',
                                            'left alone'])

    # ---- helpers --------------------------------------------------------
    def stop(self):
        self.cmd_pub.publish(Twist())

    def spin_blade(self, speed):
        self.blade_pub.publish(Float64(data=float(speed)))

    def to_world(self, bx, by):
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        return x + c * bx - s * by, y + s * bx + c * by

    def save_img(self, msg, path):
        if msg is None or msg.encoding != 'rgb8':
            return False
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        rgb = arr[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return True

    def take_photo(self, tag):
        x, y, _ = self.pose
        name = self.out_dir / f'{self.photo_count:03d}_{tag}_x{x:+.2f}_y{y:+.2f}.png'
        if not self.save_img(self.front_img, name):
            self.get_logger().warn('No usable front-camera image yet; skipping photo')
            return
        self.photo_count += 1
        self.last_photo_xy = (x, y)
        self.get_logger().info(f'Saved {name.name}')

    def save_weed_photos(self, tag):
        n = self.weed_id
        ok1 = self.save_img(self.ground_img, self.weed_dir / f'weed{n:02d}_{tag}_raw.png')
        ok2 = self.save_img(self.annot_img, self.weed_dir / f'weed{n:02d}_{tag}_annotated.png')
        if ok1 or ok2:
            self.get_logger().info(f'Saved weed {n} "{tag}" photos')

    def pick_weed(self):
        """Closest detection that is inside the area and not already cut (odom frame)."""
        if self.pose is None or self.now() - self.det_rx > 0.5:
            return None
        best = None
        for bx, by in self.dets:
            wx, wy = self.to_world(bx, by)
            if not (self.x_min - 0.3 <= wx <= self.x_max + 0.3 and
                    self.y_min - 0.3 <= wy <= self.y_max + 0.3):
                continue
            if any(math.hypot(wx - hx, wy - hy) < self.handled_radius
                   for hx, hy in self.handled):
                continue
            near = next((c for c in self.crops_seen
                         if math.hypot(wx - c[0], wy - c[1]) < self.crop_clearance), None)
            if near is not None:                  # too close to a crop: the blade could hit it
                key = (round(wx, 1), round(wy, 1))
                if key not in self.warned_near_crop:
                    self.warned_near_crop.add(key)
                    self.crops_skipped += 1
                    self.get_logger().warn(
                        f'Weed at ({wx:.2f}, {wy:.2f}) is within {self.crop_clearance} m of a '
                        f'crop - NOT cutting it, to protect the crop')
                continue
            d = math.hypot(bx, by)
            if best is None or d < best[0]:
                best = (d, wx, wy)
        return None if best is None else (best[1], best[2])

    def set_mode(self, mode):
        self.mode = mode
        self.mode_t0 = self.now()

    def resume(self):
        self.set_mode('PATROL')
        self.state = 'TURN'

    # ---- weed removal (the simulated "cut") -------------------------------
    def delete_entity(self, name):
        if self.del_cli is not None and self.del_cli.service_is_ready():
            req = DeleteEntity.Request()
            req.entity = Entity()
            req.entity.name = name
            req.entity.type = Entity.MODEL
            self.del_cli.call_async(req)
        else:
            subprocess.Popen(
                ['gz', 'service', '-s', f'/world/{self.world}/remove',
                 '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '2000', '--req', f'name: "{name}" type: MODEL'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def remove_target(self):
        tx, ty = self.target
        best = None
        for w in self.weeds:
            if w['removed']:
                continue
            d = math.hypot(w['x'] - tx, w['y'] - ty)
            if d < self.match_radius and (best is None or d < best[0]):
                best = (d, w)
        self.handled.append((tx, ty))
        self.cut_count += 1
        for cx, cy in self.crops_truth:
            if math.hypot(cx - tx, cy - ty) < self.crop_hit_radius:
                self.crop_hits += 1
                self.get_logger().error(f'A crop at ({cx:.2f}, {cy:.2f}) was inside the blade zone!')
        name = ''
        if best:
            best[1]['removed'] = True
            name = best[1]['name']
            self.delete_entity(name)
            self.get_logger().info(f'Cut weed {self.weed_id} at ({tx:.2f}, {ty:.2f}) -> {name}')
        else:
            self.get_logger().warn(
                f'Cut weed {self.weed_id} at ({tx:.2f}, {ty:.2f}) but no simulator model '
                f'within {self.match_radius} m was removed')
        with open(self.report_path, 'a', newline='') as f:
            csv.writer(f).writerow([self.weed_id, name, f'{tx:.3f}', f'{ty:.3f}',
                                    f'{self.now():.1f}'])

    # ---- weed-handling modes ----------------------------------------------
    def run_weed_mode(self):
        x, y, yaw = self.pose
        t = self.now() - self.mode_t0
        cmd = Twist()

        if self.mode == 'SETTLE':
            if t >= self.settle_time and self.det_rx >= self.mode_t0 + self.settle_time:
                w = self.pick_weed()
                if w is None:
                    self.get_logger().warn('Weed no longer visible; resuming patrol')
                    self.resume()
                    return
                self.target = w
                self.weed_id += 1
                self.get_logger().info(
                    f'Weed {self.weed_id} located at ({w[0]:.2f}, {w[1]:.2f})')
                self.save_weed_photos('found')
                self.set_mode('ALIGN')
                return
            if t > self.settle_time + 2.0:
                self.resume()
                return

        elif self.mode == 'ALIGN':
            dx, dy = self.target[0] - x, self.target[1] - y
            if math.hypot(dx, dy) < 0.15:
                self.set_mode('APPROACH')
                return
            err = wrap(math.atan2(dy, dx) - yaw)
            if abs(err) < 0.04:
                self.set_mode('APPROACH')
                return
            w = clip(1.5 * err, -self.ang, self.ang)
            cmd.angular.z = w if abs(w) >= 0.15 else math.copysign(0.15, w)
            if t > 15.0:
                self.resume()
                return

        elif self.mode == 'APPROACH':
            dx, dy = self.target[0] - x, self.target[1] - y
            remaining = math.hypot(dx, dy) - self.blade_offset   # >0: drive forward
            if abs(remaining) < 0.03:
                self.stop()
                self.spin_blade(self.blade_speed)
                self.removed = False
                self.set_mode('CUT')
                return
            v = clip(remaining, -0.12, 0.25)
            cmd.linear.x = v if abs(v) >= 0.04 else math.copysign(0.04, v)
            if v > 0 and math.hypot(dx, dy) > 0.15:
                cmd.angular.z = clip(2.0 * wrap(math.atan2(dy, dx) - yaw), -0.4, 0.4)
            if t > 15.0:
                self.get_logger().warn('Approach timed out; skipping this weed')
                self.handled.append(self.target)
                self.resume()
                return

        elif self.mode == 'CUT':
            self.spin_blade(self.blade_speed)
            if not self.removed and t >= 0.5 * self.cut_duration:
                self.remove_target()
                self.removed = True
            if t >= self.cut_duration:
                self.spin_blade(0.0)
                self.set_mode('POST')
                return

        elif self.mode == 'POST':
            if t >= 0.7:
                self.save_weed_photos('after')
                self.resume()
                return

        self.cmd_pub.publish(cmd)

    def write_summary(self):
        weeds_total = len(self.weeds)
        weeds_cut = sum(1 for w in self.weeds if w['removed'])
        crops_total = len(self.crops_truth)
        crops_found = sum(1 for tx, ty in self.crops_truth
                          if any(math.hypot(tx - c[0], ty - c[1]) < 0.25 for c in self.crops_seen))
        lines = [
            '=== Weed-Zap run summary ===',
            f'Weeds cut:            {weeds_cut} of {weeds_total}',
            f'Crops recognised:     {crops_found} of {crops_total}' if crops_total else
            f'Crops recognised:     {len(self.crops_seen)}',
            f'Crops damaged:        {self.crop_hits}   (must be 0)',
            f'Weeds skipped (too close to a crop): {self.crops_skipped}',
        ]
        for ln in lines:
            self.get_logger().info(ln)
        (self.out_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')

    # ---- main control loop --------------------------------------------------
    def step(self):
        if self.pose is None or self.finished:
            return
        x, y, yaw = self.pose

        # Geofence: never leave the specified area (plus a small margin)
        m = self.margin
        if not (self.x_min - m <= x <= self.x_max + m and
                self.y_min - m <= y <= self.y_max + m):
            self.stop()
            self.spin_blade(0.0)
            self.get_logger().error('Outside geofence! Stopping.')
            self.finished = True
            return

        if self.mode != 'PATROL':
            self.run_weed_mode()
            return

        if self.cut_weeds and self.pick_weed() is not None:
            self.stop()
            self.get_logger().info('Weed spotted - stopping to confirm')
            self.set_mode('SETTLE')
            return

        if self.idx >= len(self.waypoints):
            self.stop()
            self.take_photo('final')
            self.get_logger().info(
                f'Patrol complete: {self.photo_count} lane photos, '
                f'{self.weed_id} weeds found, {self.cut_count} cut. Report: {self.report_path}')
            self.write_summary()
            self.finished = True
            return

        tx, ty = self.waypoints[self.idx]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        err = wrap(math.atan2(dy, dx) - yaw)
        cmd = Twist()

        if dist < 0.10:                       # reached waypoint
            self.stop()
            self.take_photo(f'wp{self.idx}')
            self.idx += 1
            self.state = 'TURN'
            return

        if self.state == 'TURN':
            if abs(err) < 0.05:
                self.state = 'DRIVE'
            else:
                w = clip(1.5 * err, -self.ang, self.ang)
                cmd.angular.z = w if abs(w) >= 0.15 else math.copysign(0.15, w)
        else:  # DRIVE
            if abs(err) > 0.5:
                self.state = 'TURN'
            else:
                cmd.linear.x = min(self.lin, 0.8 * dist + 0.05)
                cmd.angular.z = clip(2.0 * err, -self.ang, self.ang)
                if self.last_photo_xy is None:
                    self.last_photo_xy = (x, y)
                if math.hypot(x - self.last_photo_xy[0],
                              y - self.last_photo_xy[1]) >= self.photo_dist:
                    self.take_photo('lane')

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = AreaPatrol()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.spin_blade(0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
