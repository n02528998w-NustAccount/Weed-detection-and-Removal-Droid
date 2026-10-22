#!/usr/bin/env python3
"""Add crop plants (cabbages) to worlds/arena.sdf and write config/crops.yaml.

Run from the package folder:   python3 tools/generate_crops.py
Safe to re-run: it removes any previously generated crops first.

Layout: a row of cabbages beside each patrol lane (y = -3, -1, 1, 3), 0.24 m to
one side so they sit inside the ground-camera view but clear of the wheels.
A few extra crops are planted right next to weeds as "tricky" cases: the robot
must cut the weed and leave the neighbouring crop untouched.
"""
import math
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SDF = ROOT / 'worlds' / 'arena.sdf'
LANES = [-3.0, -1.0, 1.0, 3.0]
SIDE = 0.24            # lateral offset of crop rows from the lane (m)
SPACING = 0.9          # distance between crops along a row (m)
MIN_WEED_DIST = 0.30   # a crop is never planted closer than this to a weed
MIN_CROP_DIST = 0.30
NEIGHBOUR_DX = 0.32    # "tricky" crop distance from a weed, along the lane
TRICKY_WEEDS = ['weed_2', 'weed_5', 'weed_8', 'weed_11']

LEAF = '0.05 0.40 0.28 1'     # dark blue-green  (hue ~79 in OpenCV)
HEAD = '0.08 0.55 0.38 1'     # same hue, lighter


def mat(c):
    return f'<material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>'


def crop_model(name, x, y, yaw):
    v = [f'''        <visual name="head">
          <pose>0 0 0.04 0 0 0</pose>
          <geometry><ellipsoid><radii>0.035 0.035 0.03</radii></ellipsoid></geometry>
          {mat(HEAD)}
        </visual>''']
    for i in range(6):
        a = i * math.pi / 3
        v.append(f'''        <visual name="leaf{i}">
          <pose>{0.022 * math.cos(a):.4f} {0.022 * math.sin(a):.4f} 0.022 0 -0.3 {a:.4f}</pose>
          <geometry><ellipsoid><radii>0.025 0.018 0.008</radii></ellipsoid></geometry>
          {mat(LEAF)}
        </visual>''')
    body = '\n'.join(v)
    return f'''    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.3f}</pose>
      <link name="link">
{body}
      </link>
    </model>
'''


def main():
    text = SDF.read_text()
    # drop crops from an earlier run
    text = re.sub(r'\n?\s*<!-- Crops:.*?<!-- End crops -->\n', '\n', text, flags=re.S)

    weeds = {}
    for m in re.finditer(r'<model name="(weed_\d+)">\s*<static>true</static>\s*'
                         r'<pose>([-\d.]+) ([-\d.]+)', text):
        weeds[m.group(1)] = (float(m.group(2)), float(m.group(3)))

    crops = []

    def ok(x, y, others):
        return (all(math.hypot(x - wx, y - wy) >= MIN_WEED_DIST for wx, wy in weeds.values())
                and all(math.hypot(x - ox, y - oy) >= MIN_CROP_DIST for ox, oy in others))

    # tricky crops first, so the regular rows are the ones that make way
    for wn in TRICKY_WEEDS:
        wx, wy = weeds[wn]
        lane = min(LANES, key=lambda l: abs(l - wy))
        side = -1 if wy < lane else 1
        for dx in (NEIGHBOUR_DX, -NEIGHBOUR_DX):
            c = (wx + dx, lane + side * SIDE)
            if ok(c[0], c[1], crops) and -4.0 <= c[0] <= 4.0:
                crops.append(c)
                break
    n_tricky = len(crops)
    for lane in LANES:
        for i in range(9):
            c = (-3.6 + i * SPACING, lane + SIDE)
            if ok(c[0], c[1], crops):
                crops.append(c)

    block = ['    <!-- Crops: cabbages the robot must LEAVE ALONE (visual-only, like the weeds) -->\n']
    for i, (x, y) in enumerate(crops, 1):
        block.append(crop_model(f'crop_{i}', x, y, (i * 0.9) % 6.28))
    block.append('    <!-- End crops -->\n')
    text = text.replace('  </world>', ''.join(block) + '  </world>')
    SDF.write_text(text)

    with open(ROOT / 'config' / 'crops.yaml', 'w') as f:
        f.write('# Simulator bookkeeping ONLY (ground truth for the score at the end of a run).\n'
                '# The detector never reads this: crops are recognised from the camera image.\n')
        for i, (x, y) in enumerate(crops, 1):
            f.write(f'- {{name: crop_{i}, x: {x:.3f}, y: {y:.3f}}}\n')
    print(f'{len(crops)} crops written ({n_tricky} placed next to weeds); {len(weeds)} weeds kept.')


if __name__ == '__main__':
    main()
