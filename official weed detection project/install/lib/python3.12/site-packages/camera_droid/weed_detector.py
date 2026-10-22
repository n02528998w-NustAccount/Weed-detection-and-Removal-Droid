#!/usr/bin/env python3
"""Tell weeds from crops in the downward-facing ground camera image.

Publishes
  weeds/detections  geometry_msgs/PoseArray  WEED centres in base_link (metres)  -> may be cut
  crops/detections  geometry_msgs/PoseArray  CROP centres in base_link (metres)  -> never cut
  weeds/annotated   sensor_msgs/Image        image with boxes: red = weed, cyan = crop

Geometry: the ground camera looks straight down, image "up" is robot-forward
and image "right" is robot-right. With a pinhole model, one pixel covers
(camera_z - plant_height) / f metres on the plane of the plant.
"""
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from camera_droid.plant_classifier import classify_plants


class WeedDetector(Node):
    def __init__(self):
        super().__init__('weed_detector')
        self.declare_parameter('camera_x', 0.30)      # camera position in base_link (m)
        self.declare_parameter('camera_z', 0.38)
        self.declare_parameter('weed_height', 0.06)   # typical height of a weed's centre (m)
        self.declare_parameter('crop_height', 0.05)   # typical height of a crop's centre (m)
        self.declare_parameter('hfov', 1.5708)
        self.declare_parameter('min_area', 600.0)     # pixels
        self.declare_parameter('hsv_lower', [35, 90, 40])        # weed: OpenCV HSV, H 0-180
        self.declare_parameter('hsv_upper', [64, 255, 255])
        self.declare_parameter('crop_hsv_lower', [70, 60, 25])   # crop: dark blue-green
        self.declare_parameter('crop_hsv_upper', [100, 255, 255])

        gp = self.get_parameter
        self.cam_x = gp('camera_x').value
        self.cam_z = gp('camera_z').value
        self.weed_h = gp('weed_height').value
        self.crop_h = gp('crop_height').value
        self.hfov = gp('hfov').value
        self.min_area = gp('min_area').value
        self.wlo, self.whi = gp('hsv_lower').value, gp('hsv_upper').value
        self.clo, self.chi = gp('crop_hsv_lower').value, gp('crop_hsv_upper').value

        self.weed_pub = self.create_publisher(PoseArray, 'weeds/detections', 10)
        self.crop_pub = self.create_publisher(PoseArray, 'crops/detections', 10)
        self.img_pub = self.create_publisher(Image, 'weeds/annotated', 10)
        self.create_subscription(Image, 'ground_camera/image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.warned = False
        self.get_logger().info('Plant detector ready: light-green = WEED (cut), '
                               'dark blue-green = CROP (leave alone).')

    def to_poses(self, blobs, header, w, h, plant_h):
        f = (w / 2.0) / math.tan(self.hfov / 2.0)            # focal length in pixels
        scale = (self.cam_z - plant_h) / f                   # metres per pixel at plant height
        poses = PoseArray()
        poses.header = header
        poses.header.frame_id = 'base_link'
        for b in blobs:
            p = Pose()
            p.position.x = self.cam_x + (h / 2.0 - b['cy']) * scale   # forward
            p.position.y = (w / 2.0 - b['cx']) * scale                # left
            p.orientation.w = 1.0
            poses.poses.append(p)
            b['xy'] = (p.position.x, p.position.y)
        return poses

    def on_image(self, msg):
        if msg.encoding != 'rgb8':
            if not self.warned:
                self.get_logger().warn(f'Unsupported encoding {msg.encoding}; need rgb8')
                self.warned = True
            return
        w, h = msg.width, msg.height
        arr = np.frombuffer(msg.data, np.uint8).reshape(h, msg.step)
        rgb = arr[:, :w * 3].reshape(h, w, 3).copy()   # writable copy so we can draw on it

        weeds, crops = classify_plants(rgb, self.min_area, self.wlo, self.whi,
                                       self.clo, self.chi)

        # crops first, so the patrol node always knows about them before it reacts to a weed
        self.crop_pub.publish(self.to_poses(crops, msg.header, w, h, self.crop_h))
        self.weed_pub.publish(self.to_poses(weeds, msg.header, w, h, self.weed_h))

        for b in weeds:
            x, y, bw, bh = b['box']
            cv2.rectangle(rgb, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(rgb, f'WEED cut ({b["xy"][0]:.2f}, {b["xy"][1]:.2f})',
                        (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        for b in crops:
            x, y, bw, bh = b['box']
            cv2.rectangle(rgb, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            cv2.putText(rgb, f'CROP leave ({b["xy"][0]:.2f}, {b["xy"][1]:.2f})',
                        (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
