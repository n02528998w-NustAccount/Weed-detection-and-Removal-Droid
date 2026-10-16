#!/usr/bin/env python3
"""Find green weeds in the downward-facing ground camera image.

Publishes
  weeds/detections  geometry_msgs/PoseArray  weed centres in the base_link frame (metres)
  weeds/annotated   sensor_msgs/Image        camera image with the detections drawn on it

Geometry: the ground camera looks straight down, image "up" is robot-forward
and image "right" is robot-right. With a pinhole model, one pixel covers
(camera_z - weed_height) / f metres on the plane of the weed.
"""
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class WeedDetector(Node):
    def __init__(self):
        super().__init__('weed_detector')
        self.declare_parameter('camera_x', 0.30)      # camera position in base_link (m)
        self.declare_parameter('camera_z', 0.38)
        self.declare_parameter('weed_height', 0.06)   # typical height of a weed's centre (m)
        self.declare_parameter('hfov', 1.5708)
        self.declare_parameter('min_area', 600.0)     # pixels
        self.declare_parameter('hsv_lower', [35, 90, 40])   # OpenCV HSV: H 0-180
        self.declare_parameter('hsv_upper', [85, 255, 255])

        gp = self.get_parameter
        self.cam_x = gp('camera_x').value
        self.cam_z = gp('camera_z').value
        self.weed_h = gp('weed_height').value
        self.hfov = gp('hfov').value
        self.min_area = gp('min_area').value
        self.lo = np.array(gp('hsv_lower').value, dtype=np.uint8)
        self.hi = np.array(gp('hsv_upper').value, dtype=np.uint8)

        self.det_pub = self.create_publisher(PoseArray, 'weeds/detections', 10)
        self.img_pub = self.create_publisher(Image, 'weeds/annotated', 10)
        self.create_subscription(Image, 'ground_camera/image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.warned = False
        self.get_logger().info('Weed detector ready (looking for green blobs).')

    def on_image(self, msg):
        if msg.encoding != 'rgb8':
            if not self.warned:
                self.get_logger().warn(f'Unsupported encoding {msg.encoding}; need rgb8')
                self.warned = True
            return
        w, h = msg.width, msg.height
        arr = np.frombuffer(msg.data, np.uint8).reshape(h, msg.step)
        rgb = arr[:, :w * 3].reshape(h, w, 3).copy()   # writable copy so we can draw on it

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lo, self.hi)
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        f = (w / 2.0) / math.tan(self.hfov / 2.0)           # focal length in pixels
        scale = (self.cam_z - self.weed_h) / f              # metres per pixel at weed height

        poses = PoseArray()
        poses.header = msg.header
        poses.header.frame_id = 'base_link'

        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue                                    # partly out of view: wait for a full blob
            m = cv2.moments(c)
            cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
            p = Pose()
            p.position.x = self.cam_x + (h / 2.0 - cy) * scale   # forward
            p.position.y = (w / 2.0 - cx) * scale                # left
            p.orientation.w = 1.0
            poses.poses.append(p)
            cv2.rectangle(rgb, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(rgb, f'weed ({p.position.x:.2f}, {p.position.y:.2f})',
                        (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        self.det_pub.publish(poses)          # always published (empty = nothing seen)

        out = Image()
        out.header = msg.header
        out.height, out.width = h, w
        out.encoding = 'rgb8'
        out.step = w * 3
        out.data = rgb.tobytes()
        self.img_pub.publish(out)


def main():
    rclpy.init()
    node = WeedDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
