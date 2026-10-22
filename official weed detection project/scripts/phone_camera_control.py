#!/usr/bin/env python3
"""
Phone-camera hand-gesture controller for the camera_droid robot.

Phone streams video (Android "IP Webcam": http://<phone-ip>:8080/video,
DroidCam: http://<phone-ip>:4747/video, or an integer for a local webcam).

  RIGHT hand  (appears on the right of the preview)  -> WHEELS
      open palm : drive. Hand height  -> forward/back speed
                         Hand left/right -> turning
      fist      : stop wheels
  LEFT hand   -> WEED CUTTER
      open palm : blade ON      fist : blade OFF
  No hand for `timeout` s -> everything stops (fail-safe).
  Keys in the preview window: q = quit.

Publishes: geometry_msgs/Twist on `cmd_vel`, std_msgs/Float64 on `blade/spin`
"""
import threading
import time

import cv2
import mediapipe as mp
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64

TIPS, PIPS = [8, 12, 16, 20], [6, 10, 14, 18]


def fingers_up(lm):
    """Count extended fingers (index..pinky) using tip-above-pip test."""
    return sum(lm[t].y < lm[p].y for t, p in zip(TIPS, PIPS))


class StreamReader(threading.Thread):
    """Always keeps only the newest frame (avoids network-stream lag)."""

    def __init__(self, src):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        self.frame, self.running = None, True

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame = f
            else:
                time.sleep(0.05)


class CameraControl(Node):
    def __init__(self):
        super().__init__('phone_camera_control')
        self.declare_parameter('stream_url', 'http://192.168.1.50:8080/video')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('blade_topic', 'blade/spin')
        self.declare_parameter('max_linear', 0.4)     # m/s
        self.declare_parameter('max_angular', 1.2)    # rad/s
        self.declare_parameter('blade_speed', 40.0)   # rad/s
        self.declare_parameter('deadzone', 0.12)      # fraction of half-frame
        self.declare_parameter('timeout', 0.5)        # s without hand -> stop
        self.declare_parameter('show_window', True)
        g = lambda n: self.get_parameter(n).value
        self.p = {n: g(n) for n in ['max_linear', 'max_angular', 'blade_speed',
                                    'deadzone', 'timeout', 'show_window']}

        self.vel_pub = self.create_publisher(Twist, g('cmd_vel_topic'), 10)
        self.blade_pub = self.create_publisher(Float64, g('blade_topic'), 10)

        self.reader = StreamReader(g('stream_url'))
        self.reader.start()
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.draw = mp.solutions.drawing_utils

        self.lin = self.ang = self.blade = 0.0
        self.last_seen = time.time()
        self.create_timer(1 / 20.0, self.step)
        self.get_logger().info(f"Streaming from {g('stream_url')}")

    def dz(self, v):
        d = self.p['deadzone']
        return 0.0 if abs(v) < d else (abs(v) - d) / (1 - d) * (1 if v > 0 else -1)

    def step(self):
        frame = self.reader.frame
        if frame is None:
            return self.publish(0.0, 0.0, 0.0)
        frame = cv2.flip(frame, 1)                     # mirror: natural left/right
        res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        tgt_lin = tgt_ang = 0.0
        blade_on = False
        seen = False
        if res.multi_hand_landmarks:
            for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                seen = True
                lm = lms.landmark
                n = fingers_up(lm)
                # MediaPipe assumes a mirrored image -> label matches what the user sees
                label = hd.classification[0].label
                if label == 'Right':                   # wheels
                    if n >= 4:
                        cx, cy = lm[9].x, lm[9].y      # palm centre
                        tgt_lin = self.dz((0.5 - cy) * 2) * self.p['max_linear']
                        tgt_ang = self.dz((0.5 - cx) * 2) * self.p['max_angular']
                else:                                  # cutter
                    blade_on = n >= 4
                self.draw.draw_landmarks(frame, lms, mp.solutions.hands.HAND_CONNECTIONS)
        if seen:
            self.last_seen = time.time()
        elif time.time() - self.last_seen > self.p['timeout']:
            tgt_lin = tgt_ang = 0.0
            blade_on = False

        # light smoothing
        a = 0.35
        self.lin += a * (tgt_lin - self.lin)
        self.ang += a * (tgt_ang - self.ang)
        self.blade = self.p['blade_speed'] if blade_on else 0.0
        self.publish(self.lin, self.ang, self.blade)

        if self.p['show_window']:
            h, w = frame.shape[:2]
            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
            cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1)
            txt = f"lin {self.lin:+.2f}  ang {self.ang:+.2f}  blade {'ON' if self.blade else 'off'}"
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('phone_camera_control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()

    def publish(self, lin, ang, blade):
        t = Twist()
        t.linear.x, t.angular.z = float(lin), float(ang)
        self.vel_pub.publish(t)
        self.blade_pub.publish(Float64(data=float(blade)))

    def destroy_node(self):
        self.reader.running = False
        try:
            self.publish(0.0, 0.0, 0.0)
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
