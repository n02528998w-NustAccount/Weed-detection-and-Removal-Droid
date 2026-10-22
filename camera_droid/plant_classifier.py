"""Crop-vs-weed classification for the downward-facing ground camera.

No ROS imports here, so it can be unit-tested on plain images.

How it decides (simple colour + size classifier, the "throw-away prototype"
version of the AI model):
  * WEEDS are bright, light green        (OpenCV hue  35-64)
  * CROPS are dark, blue-green (cabbage) (OpenCV hue  70-100)
  * Hue 65-69 is a deliberate gap: anything that is not clearly a weed is
    NOT cut ("when in doubt, leave it alone").
Hue is used because it stays almost constant when a leaf is in sun or shade.
"""
import cv2
import numpy as np

WEED_LO, WEED_HI = (35, 90, 40), (64, 255, 255)
CROP_LO, CROP_HI = (70, 60, 25), (100, 255, 255)


def _blobs(mask, min_area, w, h):
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue                       # partly out of view: wait for a full blob
        m = cv2.moments(c)
        out.append(dict(box=(x, y, bw, bh), cx=m['m10'] / m['m00'], cy=m['m01'] / m['m00']))
    return out


def classify_plants(rgb, min_area=600.0, weed_lo=WEED_LO, weed_hi=WEED_HI,
                    crop_lo=CROP_LO, crop_hi=CROP_HI):
    """Return (weeds, crops): lists of dicts with box=(x,y,w,h), cx, cy in pixels."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    weeds = _blobs(cv2.inRange(hsv, np.array(weed_lo, np.uint8), np.array(weed_hi, np.uint8)),
                   min_area, w, h)
    crops = _blobs(cv2.inRange(hsv, np.array(crop_lo, np.uint8), np.array(crop_hi, np.uint8)),
                   min_area, w, h)
    return weeds, crops
