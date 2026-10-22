#!/usr/bin/env python3
"""Build worlds/arena.sdf from the models in models/ (replaces tools/generate_crops.py).

Run from the package folder (after tools/generate_assets.py):

    python3 tools/build_world.py

Writes
  worlds/arena.sdf     soil ground, hedge/fence walls, trees, 11 blackjack weeds, 37 groundnut crops
  config/weeds.yaml    weed names + positions (the patrol node deletes a weed model by name when cut)
  config/crops.yaml    ground truth for the end-of-run score

Everything is placed with <include><uri>model://...</uri>, so the world file stays small and all
geometry lives in models/ as the lecturer asked. Weed/crop names (weed_N, crop_N) and positions
are the same as before, so the patrol, detector and scoring code keep working unchanged.
Safe to re-run; the old arena.sdf / yaml files are saved once as *.bak.
"""
import math
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDF = ROOT / 'worlds' / 'arena.sdf'

# ---- layout (same as the original arena) ---------------------------------------------
WEEDS = {1: (-2.5, -3.1), 2: (-0.5, -2.9), 3: (1.5, -3.05), 4: (-3.0, -0.95), 5: (0.5, -1.1),
         6: (2.2, -0.9), 7: (-1.5, 1.1), 8: (1.2, 0.9), 9: (3.5, 1.05), 10: (-2.0, 2.95),
         11: (2.8, 3.0)}
LANES = [-3.0, -1.0, 1.0, 3.0]
SIDE, SPACING = 0.24, 0.9
MIN_WEED_DIST = MIN_CROP_DIST = 0.30
NEIGHBOUR_DX = 0.32
TRICKY_WEEDS = ['weed_2', 'weed_5', 'weed_8', 'weed_11']

# ---- look ------------------------------------------------------------------------------
# which model forms each wall of the 10 m x 10 m arena (hedge_wall or wooden_fence)
WALLS = {'n': 'hedge_wall', 's': 'hedge_wall', 'e': 'wooden_fence', 'w': 'wooden_fence'}
N_TREES = 14
CROP_MODELS = ['groundnuts', 'groundnuts_2', 'groundnuts_3']
WEED_MODELS = ['blackjack', 'blackjack_2', 'blackjack_3']
SEED = 7

HEADER = '''<?xml version="1.0"?>
<sdf version="1.9">
  <world name="arena">
    <physics name="default_physics" type="ode">
      <max_step_size>0.003</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <scene>
      <ambient>0.45 0.45 0.45 1</ambient>
      <background>0.62 0.78 0.92 1</background>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.88 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.2 -0.9</direction>
    </light>
'''

# the original photo targets, unchanged
TARGETS = '''
    <!-- Things to photograph (kept clear of the patrol lanes at y = -3, -1, 1, 3) -->
    <model name="red_box">
      <static>true</static><pose>0 2 0.25 0 0 0.4</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>0.5 0.5 0.5</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>0.5 0.5 0.5</size></box></geometry>
          <material><ambient>0.9 0.1 0.1 1</ambient><diffuse>0.9 0.1 0.1 1</diffuse></material></visual>
      </link>
    </model>
    <model name="orange_box">
      <static>true</static><pose>0 -2 0.25 0 0 -0.3</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>0.5 0.5 0.5</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>0.5 0.5 0.5</size></box></geometry>
          <material><ambient>0.95 0.5 0.1 1</ambient><diffuse>0.95 0.5 0.1 1</diffuse></material></visual>
      </link>
    </model>
    <model name="blue_cylinder">
      <static>true</static><pose>3 0 0.4 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><cylinder><radius>0.3</radius><length>0.8</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.3</radius><length>0.8</length></cylinder></geometry>
          <material><ambient>0.1 0.3 0.8 1</ambient><diffuse>0.1 0.3 0.8 1</diffuse></material></visual>
      </link>
    </model>
    <model name="purple_cylinder">
      <static>true</static><pose>-3 0 0.4 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><cylinder><radius>0.3</radius><length>0.8</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.3</radius><length>0.8</length></cylinder></geometry>
          <material><ambient>0.6 0.2 0.8 1</ambient><diffuse>0.6 0.2 0.8 1</diffuse></material></visual>
      </link>
    </model>
'''


def include(model, name, x, y, z=0.0, yaw=0.0):
    return (f'    <include>\n      <uri>model://{model}</uri>\n      <name>{name}</name>\n'
            f'      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>\n    </include>\n')


def walls():
    out = ['\n    <!-- Walls: 2 m segments from models/, inner face at 5.0 m like the old walls -->\n']
    seg = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]               # segment centres along a wall
    for side, model in WALLS.items():
        # hedge is 0.5 m thick (centre 0.25 outside the line), fence is ~0.1 m thick
        off = 5.25 if model == 'hedge_wall' else 5.05
        centres = seg if side in 'ns' else [-4.0, -2.0, 0.0, 2.0, 4.0]
        for i, c in enumerate(centres, 1):
            if side == 'n':
                x, y, yaw = c, off, 0.0
            elif side == 's':
                x, y, yaw = c, -off, math.pi
            elif side == 'e':
                x, y, yaw = off, c, math.pi / 2
            else:
                x, y, yaw = -off, c, -math.pi / 2
            out.append(include(model, f'wall_{side}_{i}', x, y, 0.0, yaw))
    return ''.join(out)


def trees(rnd):
    out = ['\n    <!-- Trees around the outside of the arena -->\n']
    placed = []
    tries = 0
    while len(placed) < N_TREES and tries < 5000:
        tries += 1
        x, y = rnd.uniform(-10.5, 10.5), rnd.uniform(-10.5, 10.5)
        if max(abs(x), abs(y)) < 7.5:                      # keep clear of the walls
            continue
        if any(math.hypot(x - px, y - py) < 3.0 for px, py in placed):
            continue
        placed.append((x, y))
    for i, (x, y) in enumerate(placed, 1):
        model = 'tree_oak' if rnd.random() < 0.6 else 'tree_acacia'
        out.append(include(model, f'tree_{i}', x, y, 0.0, rnd.uniform(0, 6.28)))
    return ''.join(out)


def plan_crops():
    """Same placement logic as the old generate_crops.py."""
    weeds = {f'weed_{k}': v for k, v in WEEDS.items()}
    crops = []

    def ok(x, y):
        return (all(math.hypot(x - wx, y - wy) >= MIN_WEED_DIST for wx, wy in weeds.values())
                and all(math.hypot(x - ox, y - oy) >= MIN_CROP_DIST for ox, oy in crops))

    for wn in TRICKY_WEEDS:                                # crops right next to weeds first
        wx, wy = weeds[wn]
        lane = min(LANES, key=lambda l: abs(l - wy))
        side = -1 if wy < lane else 1
        for dx in (NEIGHBOUR_DX, -NEIGHBOUR_DX):
            c = (wx + dx, lane + side * SIDE)
            if ok(*c) and -4.0 <= c[0] <= 4.0:
                crops.append(c)
                break
    n_tricky = len(crops)
    for lane in LANES:
        for i in range(9):
            c = (-3.6 + i * SPACING, lane + SIDE)
            if ok(*c):
                crops.append(c)
    return crops, n_tricky


def backup(path):
    bak = Path(str(path) + '.bak')
    if path.exists() and not bak.exists():
        shutil.copy(path, bak)


def main():
    for m in ['soil_ground', 'hedge_wall', 'wooden_fence', 'tree_oak', 'tree_acacia',
              CROP_MODELS[0], WEED_MODELS[0]]:
        if not (ROOT / 'models' / m / 'model.sdf').exists():
            raise SystemExit(f'models/{m} is missing - run  python3 tools/generate_assets.py  first.')

    rnd = random.Random(SEED)
    crops, n_tricky = plan_crops()

    body = [HEADER,
            '\n    <!-- Ground -->\n', include('soil_ground', 'ground', 0, 0),
            walls(), trees(rnd), TARGETS,
            '\n    <!-- Weeds: blackjack (kept at the same names/positions the patrol node expects) -->\n']
    for k, (x, y) in WEEDS.items():
        body.append(include(rnd.choice(WEED_MODELS), f'weed_{k}', x, y, 0.0, rnd.uniform(0, 6.28)))
    body.append('\n    <!-- Crops: groundnuts the robot must LEAVE ALONE -->\n')
    for i, (x, y) in enumerate(crops, 1):
        body.append(include(rnd.choice(CROP_MODELS), f'crop_{i}', x, y, 0.0, rnd.uniform(0, 6.28)))
    body.append('  </world>\n</sdf>\n')

    backup(SDF)
    SDF.write_text(''.join(body))

    (ROOT / 'config').mkdir(exist_ok=True)
    for name, rows in (('weeds', [(f'weed_{k}', x, y) for k, (x, y) in WEEDS.items()]),
                       ('crops', [(f'crop_{i}', x, y) for i, (x, y) in enumerate(crops, 1)])):
        path = ROOT / 'config' / f'{name}.yaml'
        backup(path)
        with open(path, 'w') as f:
            if name == 'crops':
                f.write('# Simulator bookkeeping ONLY (ground truth for the score at the end of a run).\n'
                        '# The detector never reads this: crops are recognised from the camera image.\n')
            for n, x, y in rows:
                f.write(f'- {{name: {n}, x: {x:.3f}, y: {y:.3f}}}\n')
    print(f'worlds/arena.sdf written: {len(WEEDS)} weeds, {len(crops)} crops '
          f'({n_tricky} next to weeds), {N_TREES} trees, walls {WALLS}')


if __name__ == '__main__':
    main()
