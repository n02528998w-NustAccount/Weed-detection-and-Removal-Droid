#!/usr/bin/env python3
"""Generate realistic (low-poly, textured) models for the Weed-Zap arena.

Run from the package folder:   python3 tools/generate_assets.py
Needs numpy + opencv (already used by the package). Safe to re-run.

Follows the lecturer's layout, models/<name>/model.config + model.sdf:

  soil_ground     tileable PBR soil (albedo + normal + roughness), collision plane
  hedge_wall      2 m hedge segment (1.0 m high, 0.5 m thick), lumpy foliage
  wooden_fence    2 m plank fence segment (0.9 m high)
  tree_oak        round-canopy tree (~3.6 m)
  tree_acacia     flat-topped umbrella tree (~4 m)
  groundnuts, groundnuts_2, groundnuts_3   crop plant variants (compound leaves)
  blackjack, blackjack_2, blackjack_3      weed variants (serrated leaves, spiky seed heads)

All meshes are Z-up OBJ files generated in code, so no downloads are needed.
Plants/weeds are visual-only (like your old ones); hedges, fences, trees and the
ground have simple collision shapes.
"""
import math
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
OBJ_Y_UP = False           # set True if models show up lying on their side in Gazebo
GROUNDNUT_SCALE = 0.8      # young crop plant, ~0.2 m across (rows are only 0.24 m from the lane)


# =============================================================== textures
def fft_noise(n, beta, seed):
    """Tileable noise in [0, 1] (spectral synthesis, so it wraps seamlessly)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    r = np.sqrt(f[None, :] ** 2 + f[:, None] ** 2)
    r[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / r ** beta
    spec[0, 0] = 0
    a = np.real(np.fft.ifft2(spec))
    a -= a.min()
    return a / a.max()


def ramp(h, c0, c1):
    c0, c1 = np.array(c0, float), np.array(c1, float)
    return c0 + (c1 - c0) * h[..., None]


def save_png(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), np.ascontiguousarray(img[..., ::-1]))


def normal_map(h, strength):
    dx = np.roll(h, -1, 1) - np.roll(h, 1, 1)
    dy = np.roll(h, -1, 0) - np.roll(h, 1, 0)
    n = np.dstack([-dx * strength, -dy * strength, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return (n * 0.5 + 0.5) * 255


def soil_textures(tex_dir, n=1024):
    h = 0.5 * fft_noise(n, 1.1, 1) + 0.3 * fft_noise(n, 1.8, 2) + 0.2 * fft_noise(n, 0.5, 3)
    h = (h - h.min()) / (h.max() - h.min())
    rgb = ramp(h, (62, 43, 30), (128, 94, 66))
    rgb *= (0.88 + 0.24 * fft_noise(n, 0.3, 4))[..., None]
    img = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    height = np.ascontiguousarray(h.astype(np.float32))
    rng = np.random.default_rng(5)
    for _ in range(900):                       # pebbles and clods
        x, y = (int(v) for v in rng.integers(8, n - 8, 2))
        r = int(rng.integers(2, 6))
        shade = float(rng.uniform(0.7, 1.25))
        col = tuple(int(min(255, c * shade)) for c in (140, 124, 104))
        cv2.circle(img, (x, y), r, col, -1)
        cv2.circle(height, (x, y), r, 1.0, -1)
    height = cv2.GaussianBlur(height, (0, 0), 1.3)
    save_png(tex_dir / 'soil_albedo.png', img[..., ::-1][..., ::-1])
    save_png(tex_dir / 'soil_normal.png', normal_map(height, 8.0))
    save_png(tex_dir / 'soil_roughness.png', np.full((n, n, 3), 235))


def foliage_texture(path, seed, dark=(14, 48, 16), light=(72, 132, 40), leaves=2500, n=512):
    h = fft_noise(n, 1.4, seed)
    img = np.ascontiguousarray(ramp(h, dark, light).astype(np.uint8))
    rng = np.random.default_rng(seed + 1)
    d, l = np.array(dark, float), np.array(light, float)
    for _ in range(leaves):
        x, y = (int(v) for v in rng.integers(0, n, 2))
        ax = (int(rng.integers(5, 10)), int(rng.integers(2, 5)))
        ang = float(rng.uniform(0, 180))
        col = tuple(int(v) for v in np.clip(d + (l - d) * rng.uniform(0, 1.15), 0, 255))
        for dx in (-n, 0, n):                 # wrap so the texture still tiles
            for dy in (-n, 0, n):
                cv2.ellipse(img, (x + dx, y + dy), ax, ang, 0, 360, col, -1)
    save_png(path, img)


def leaf_texture(path, base, edge, vein, n=256, seed=0):
    v = np.linspace(0, 1, n)[:, None]
    u = np.linspace(0, 1, n)[None, :]
    h = fft_noise(n, 1.5, seed)
    centre = (1 - np.abs(u - 0.5) * 2) + 0 * v
    img = ramp(0.55 * h + 0.45 * centre, edge, base)
    mid = np.exp(-((u - 0.5) / 0.025) ** 2) + 0 * v
    side = np.exp(-((((v * 8 - np.abs(u - 0.5) * 6) % 1.0) - 0.5) / 0.07) ** 2)
    side = side * np.clip(1 - np.abs(u - 0.5) * 1.6, 0, 1)
    vm = np.clip(mid + 0.45 * side, 0, 1)[..., None]
    img = img + vm * (np.array(vein, float) - img) * 0.55
    save_png(path, img)


def streak_texture(path, seed, c0, c1, n=512, stretch=12):
    h = fft_noise(n, 1.3, seed).astype(np.float32)
    h = cv2.resize(h[:n // stretch], (n, n), interpolation=cv2.INTER_CUBIC)
    h = np.clip(0.8 * h + 0.2 * fft_noise(n, 0.6, seed + 7), 0, 1)
    h = (h - h.min()) / (h.max() - h.min())
    save_png(path, ramp(h, c0, c1))


# ===================================================================== mesh
class Mesh:
    def __init__(self):
        self.v, self.uv, self.f = [], [], []

    def add(self, p, uv):
        self.v.append(tuple(float(c) for c in p))
        self.uv.append((float(uv[0]), float(uv[1])))
        return len(self.v) - 1

    def grid(self, pts, uvs, mode='front'):
        """pts[i][j]; 'front' = normal i x j, 'flip' = opposite, 'both' = two-sided."""
        sides = {'front': (0,), 'flip': (1,), 'both': (0, 1)}[mode]
        rows, cols = len(pts), len(pts[0])
        for side in sides:
            idx = [[self.add(pts[i][j], uvs[i][j]) for j in range(cols)] for i in range(rows)]
            for i in range(rows - 1):
                for j in range(cols - 1):
                    a, b, c, d = idx[i][j], idx[i][j + 1], idx[i + 1][j + 1], idx[i + 1][j]
                    if side == 0:
                        self.f += [(a, c, b), (a, d, c)]
                    else:
                        self.f += [(a, b, c), (a, c, d)]

    def write(self, path, scale=1.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        V = np.array(self.v) * scale
        N = np.zeros_like(V)
        for a, b, c in self.f:
            fn = np.cross(V[b] - V[a], V[c] - V[a])
            N[a] += fn
            N[b] += fn
            N[c] += fn
        ln = np.linalg.norm(N, axis=1, keepdims=True)
        ln[ln == 0] = 1
        N /= ln
        if OBJ_Y_UP:                           # Z-up -> Y-up (proper rotation, winding unchanged)
            V = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1)
            N = np.stack([N[:, 0], N[:, 2], -N[:, 1]], 1)
        with open(path, 'w') as fh:
            fh.write('# generated by tools/generate_assets.py\n')
            for p in V:
                fh.write(f'v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n')
            for t in self.uv:
                fh.write(f'vt {t[0]:.5f} {t[1]:.5f}\n')
            for n in N:
                fh.write(f'vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n')
            for a, b, c in self.f:
                fh.write(f'f {a+1}/{a+1}/{a+1} {b+1}/{b+1}/{b+1} {c+1}/{c+1}/{c+1}\n')
        return len(self.f)


def frame(yaw, pitch, roll=0.0):
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    x = np.array([cp * cy, cp * sy, sp])
    y = np.array([-sy, cy, 0.0])
    z = np.cross(x, y)
    if roll:
        cr, sr = math.cos(roll), math.sin(roll)
        y, z = y * cr + z * sr, -y * sr + z * cr
    return x, y, z


def leaf(mesh, origin, yaw, pitch, length, width, roll=0.0, droop=0.25, fold=0.18,
         serrated=False, steps=5):
    x, y, z = frame(yaw, pitch, roll)
    o = np.array(origin, float)
    if serrated:
        steps *= 2
    pts, uvs = [], []
    for i in range(steps + 1):
        t = i / steps
        w = 0.5 * width * math.sin(math.pi * t ** 0.75) ** 0.9
        if serrated and 0 < i < steps and i % 2 == 0:
            w *= 0.72                          # alternate vertices give a toothed edge
        c = o + x * length * t - z * droop * length * t * t
        pts.append([c + y * j * w + z * abs(j) * w * fold for j in (-1, 0, 1)])
        uvs.append([(0.5 + 0.5 * j, t) for j in (-1, 0, 1)])
    mesh.grid(pts, uvs, 'both')


def tube(mesh, pts, radii, sides=5, vscale=1.0):
    pts = [np.array(p, float) for p in pts]
    rings, uvs, length = [], [], 0.0
    for i, p in enumerate(pts):
        if i:
            length += np.linalg.norm(p - pts[i - 1])
        t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        ref = np.array([0, 0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(t, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(t, e1)
        ang = [2 * math.pi * k / sides for k in range(sides + 1)]
        rings.append([p + radii[i] * (math.cos(a) * e1 + math.sin(a) * e2) for a in ang])
        uvs.append([(k / sides, length * vscale) for k in range(sides + 1)])
    mesh.grid(rings, uvs, 'flip')


def face_grid(mesh, o, u, v, nu, nv, tile, disp=None, off=(0.0, 0.0)):
    o, u, v = np.array(o, float), np.array(u, float), np.array(v, float)
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    pts, uvs = [], []
    for i in range(nu + 1):
        row, urow = [], []
        for j in range(nv + 1):
            p = o + u * i / nu + v * j / nv
            if disp:
                p = p + n * disp(p)
            row.append(p)
            urow.append((lu * i / nu / tile + off[0], lv * j / nv / tile + off[1]))
        pts.append(row)
        uvs.append(urow)
    mesh.grid(pts, uvs, 'front')


def box(mesh, c, size, tile=0.5, yaw=0.0, rnd=None):
    hx, hy, hz = (s / 2 for s in size)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    c = np.array(c, float)
    off = (rnd.random(), rnd.random()) if rnd else (0, 0)
    faces = [((hx, -hy, -hz), (0, 2 * hy, 0), (0, 0, 2 * hz)),
             ((-hx, hy, -hz), (0, -2 * hy, 0), (0, 0, 2 * hz)),
             ((hx, hy, -hz), (-2 * hx, 0, 0), (0, 0, 2 * hz)),
             ((-hx, -hy, -hz), (2 * hx, 0, 0), (0, 0, 2 * hz)),
             ((-hx, -hy, hz), (2 * hx, 0, 0), (0, 2 * hy, 0))]
    for o, u, v in faces:
        face_grid(mesh, c + rot @ np.array(o), rot @ np.array(u), rot @ np.array(v),
                  1, 1, tile, off=off)


def blob(mesh, centre, radii, seed, lat=8, lon=14, noise=0.2):
    rnd = random.Random(seed)
    ph = [rnd.uniform(0, 6.28) for _ in range(3)]
    c = np.array(centre, float)
    pts, uvs = [], []
    for i in range(lat + 1):
        th = math.pi * i / lat
        row, urow = [], []
        for j in range(lon + 1):
            a = 2 * math.pi * j / lon
            d = np.array([math.sin(th) * math.cos(a), math.sin(th) * math.sin(a), math.cos(th)])
            k = 1 + noise * (math.sin(3.1 * d[0] + ph[0]) * math.sin(3.3 * d[1] + ph[1])
                             + 0.5 * math.sin(5.0 * d[2] + ph[2]))
            row.append(c + d * np.array(radii) * k)
            urow.append((j / lon * 2.0, i / lat))
        pts.append(row)
        uvs.append(urow)
    mesh.grid(pts, uvs, 'front')


def make_disp(seed, amp):
    rnd = random.Random(seed)
    waves = []
    for _ in range(9):
        d = np.array([rnd.gauss(0, 1) for _ in range(3)])
        d /= np.linalg.norm(d)
        waves.append((d, rnd.uniform(7, 22), rnd.uniform(0, 6.28)))

    def f(p):
        return amp * sum(math.sin(fr * float(d @ p) + ph) for d, fr, ph in waves) / len(waves) * 2.0
    return f


# ============================================================ model writers
def write_config(d, name, desc):
    (d / 'model.config').write_text(f'''<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>camera_droid</name><email>you@example.com</email></author>
  <description>{desc}</description>
</model>
''')


def visual(name, mesh, albedo, normal=None, rough=0.9):
    nm = f'<normal_map>model://{normal}</normal_map>' if normal else ''
    return f'''      <visual name="{name}">
        <geometry><mesh><uri>model://{mesh}</uri></mesh></geometry>
        <material>
          <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0.04 0.04 0.04 1</specular>
          <pbr><metal>
            <albedo_map>model://{albedo}</albedo_map>{nm}
            <roughness>{rough}</roughness><metalness>0.0</metalness>
          </metal></pbr>
        </material>
      </visual>
'''


def write_sdf(d, name, body):
    (d / 'model.sdf').write_text(f'''<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
{body}    </link>
  </model>
</sdf>
''')


def collision(shape, mu=None):
    fr = (f'<surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>'
          if mu else '')
    return f'      <collision name="collision"><geometry>{shape}</geometry>{fr}</collision>\n'


# ================================================================== models
def make_soil():
    name, d = 'soil_ground', MODELS / 'soil_ground'
    tex = d / 'materials' / 'textures'
    soil_textures(tex)
    S = 24.0
    m = Mesh()
    face_grid(m, (-S / 2, -S / 2, 0), (S, 0, 0), (0, S, 0), 1, 1, 1.0)   # 1 tile per metre
    m.write(d / 'meshes' / 'soil.obj')
    body = visual('soil', f'{name}/meshes/soil.obj', f'{name}/materials/textures/soil_albedo.png',
                  f'{name}/materials/textures/soil_normal.png', 0.95)
    body += collision('<plane><normal>0 0 1</normal><size>60 60</size></plane>', mu=1.0)
    write_sdf(d, name, body)
    write_config(d, name, 'Tileable PBR soil ground, 24 m x 24 m')


def make_hedge():
    name, d = 'hedge_wall', MODELS / 'hedge_wall'
    L, T, H = 2.0, 0.5, 1.0
    foliage_texture(d / 'materials' / 'textures' / 'hedge.png', 11)
    disp = make_disp(3, 0.035)
    m = Mesh()
    face_grid(m, (-L / 2, -T / 2, 0), (L, 0, 0), (0, 0, H), 24, 12, 0.8, disp)
    face_grid(m, (L / 2, T / 2, 0), (-L, 0, 0), (0, 0, H), 24, 12, 0.8, disp, off=(0.3, 0))
    face_grid(m, (-L / 2, -T / 2, H), (L, 0, 0), (0, T, 0), 24, 6, 0.8, disp, off=(0.6, 0))
    face_grid(m, (L / 2, -T / 2, 0), (0, T, 0), (0, 0, H), 1, 1, 0.8)
    face_grid(m, (-L / 2, T / 2, 0), (0, -T, 0), (0, 0, H), 1, 1, 0.8)
    m.write(d / 'meshes' / 'hedge.obj')
    body = visual('hedge', f'{name}/meshes/hedge.obj', f'{name}/materials/textures/hedge.png')
    body += f'      <collision name="collision"><pose>0 0 {H/2} 0 0 0</pose><geometry><box><size>{L} {T} {H}</size></box></geometry></collision>\n'
    write_sdf(d, name, body)
    write_config(d, name, 'Hedge wall segment 2.0 x 0.5 x 1.0 m (place end to end)')


def make_fence():
    name, d = 'wooden_fence', MODELS / 'wooden_fence'
    streak_texture(d / 'materials' / 'textures' / 'wood.png', 21, (88, 66, 46), (150, 118, 84))
    rnd = random.Random(4)
    L, H = 2.0, 0.9
    m = Mesh()
    pw = 0.115
    n = int(L / (pw + 0.004))
    for i in range(n):
        h = H + rnd.uniform(-0.04, 0.03)
        x = -L / 2 + pw / 2 + i * (pw + 0.004)
        box(m, (x, 0, h / 2), (pw, 0.025, h), tile=0.4, yaw=rnd.uniform(-0.01, 0.01), rnd=rnd)
    for z in (0.25, 0.65):
        box(m, (0, 0.04, z), (L, 0.04, 0.08), tile=0.4, rnd=rnd)
    for x in (-L / 2 + 0.05, 0.0, L / 2 - 0.05):
        box(m, (x, 0.03, 0.5), (0.1, 0.1, 1.0), tile=0.4, rnd=rnd)
    m.write(d / 'meshes' / 'fence.obj')
    body = visual('fence', f'{name}/meshes/fence.obj', f'{name}/materials/textures/wood.png', rough=0.85)
    body += f'      <collision name="collision"><pose>0 0.02 {H/2} 0 0 0</pose><geometry><box><size>{L} 0.1 {H}</size></box></geometry></collision>\n'
    write_sdf(d, name, body)
    write_config(d, name, 'Wooden plank fence segment 2.0 m x 0.9 m (place end to end)')


def make_tree(name, kind, seed):
    d = MODELS / name
    tex = d / 'materials' / 'textures'
    streak_texture(tex / 'bark.png', seed, (46, 34, 26), (98, 80, 62), stretch=14)
    foliage_texture(tex / 'canopy.png', seed + 30,
                    dark=(16, 52, 18) if kind == 'oak' else (40, 78, 24),
                    light=(78, 140, 44) if kind == 'oak' else (118, 150, 52))
    rnd = random.Random(seed)
    trunk, canopy = Mesh(), Mesh()
    H = 2.6 if kind == 'oak' else 3.2
    pts, rad = [], []
    for i in range(9):
        t = i / 8
        pts.append((0.08 * math.sin(3 * t + seed), 0.06 * math.cos(2.5 * t), H * t))
        rad.append(0.16 * (1 - 0.55 * t) + 0.06 * (1 - t) ** 4)
    tube(trunk, pts, rad, sides=7, vscale=1.0)
    top = np.array(pts[-1])
    if kind == 'oak':
        for b in range(5):
            az = b * 1.25 + rnd.uniform(-0.3, 0.3)
            s = np.array(pts[5 + b % 2])
            e = s + np.array([math.cos(az) * 0.9, math.sin(az) * 0.9, 0.7])
            tube(trunk, [s, (s + e) / 2 + [0, 0, 0.1], e], [0.06, 0.04, 0.02], sides=5)
        blob(canopy, top + [0, 0, 0.5], (1.4, 1.4, 1.05), seed, 9, 16)
        for b in range(6):
            a = b * math.pi / 3 + rnd.uniform(-0.2, 0.2)
            blob(canopy, top + [math.cos(a) * 1.0, math.sin(a) * 1.0, 0.1 + rnd.uniform(0, 0.5)],
                 (0.95, 0.95, 0.75), seed + b + 1)
    else:
        for sgn in (-1, 1):
            e = top + [sgn * 0.8, 0.2 * sgn, 0.7]
            tube(trunk, [top - [0, 0, 0.3], top + [sgn * 0.3, 0, 0.2], e], [0.07, 0.055, 0.03], sides=5)
        blob(canopy, top + [0, 0, 0.7], (2.0, 2.0, 0.5), seed, 8, 18)
        for b in range(5):
            a = b * 2 * math.pi / 5 + rnd.uniform(-0.2, 0.2)
            blob(canopy, top + [math.cos(a) * 1.4, math.sin(a) * 1.4, 0.5 + rnd.uniform(0, 0.2)],
                 (1.0, 1.0, 0.36), seed + b + 1)
    trunk.write(d / 'meshes' / 'trunk.obj')
    canopy.write(d / 'meshes' / 'canopy.obj')
    body = visual('trunk', f'{name}/meshes/trunk.obj', f'{name}/materials/textures/bark.png', rough=0.95)
    body += visual('canopy', f'{name}/meshes/canopy.obj', f'{name}/materials/textures/canopy.png', rough=0.8)
    body += (f'      <collision name="collision"><pose>0 0 1.5 0 0 0</pose><geometry><cylinder>'
             f'<radius>0.2</radius><length>3.0</length></cylinder></geometry></collision>\n')
    write_sdf(d, name, body)
    write_config(d, name, 'Tree with textured trunk and canopy; trunk collision only')


# ------------------------------------------------------------------ plants
def groundnut_meshes(seed):
    rnd = random.Random(seed)
    stems, leaves = Mesh(), Mesh()

    def leafset(o, az, s):
        for off, spread in ((0.0, 0.62), (0.018 * s, 0.33)):
            for sg in (-1, 1):
                p = (o[0] + math.cos(az) * off, o[1] + math.sin(az) * off, o[2] + 0.01)
                leaf(leaves, p, az + sg * spread + rnd.uniform(-0.12, 0.12), rnd.uniform(0.15, 0.55),
                     0.048 * s * rnd.uniform(0.9, 1.1), 0.031 * s, roll=sg * rnd.uniform(0, 0.3),
                     droop=0.3, steps=4)

    for b in range(rnd.randint(8, 10)):
        az = b * 2.39996 + rnd.uniform(-0.2, 0.2)
        L, pitch = rnd.uniform(0.10, 0.15), rnd.uniform(0.7, 1.1)
        pts = []
        for k in range(5):
            t = k / 4
            r = L * t * math.cos(pitch)
            pts.append((r * math.cos(az), r * math.sin(az), 0.012 + L * math.sin(pitch) * t * (1 - 0.35 * t)))
        tube(stems, pts, [0.0035, 0.003, 0.0025, 0.002, 0.0015], sides=4)
        leafset(pts[2], az + rnd.uniform(-0.5, 0.5), 0.8)
        leafset(pts[3], az + rnd.uniform(-0.5, 0.5), 0.9)
        leafset(pts[4], az, 1.0)
    return stems, leaves, None


def blackjack_meshes(seed):
    """Young blackjack: basal rosette + short leafy stem + spiky seed heads (~0.15 m tall).
    Kept low and dense on purpose: the droid's ground camera is only 0.38 m above the soil
    and needs to see one solid green blob per weed."""
    rnd = random.Random(seed)
    stems, leaves, seeds = Mesh(), Mesh(), Mesh()
    H = rnd.uniform(0.13, 0.17)
    lx, ly = rnd.uniform(-0.02, 0.02), rnd.uniform(-0.02, 0.02)
    pts = [(lx * (k / 5) ** 2, ly * (k / 5) ** 2, H * k / 5) for k in range(6)]
    tube(stems, pts, [0.0045, 0.0042, 0.0038, 0.0034, 0.003, 0.0025], sides=4)

    def vec(az, pitch):
        return np.array([math.cos(pitch) * math.cos(az), math.cos(pitch) * math.sin(az), math.sin(pitch)])

    base = rnd.uniform(0, 6.28)
    o0 = np.array(pts[0]) + [0, 0, 0.012]
    for i in range(6):                                  # basal rosette
        az = base + i * 1.0472 + rnd.uniform(-0.2, 0.2)
        e = o0 + vec(az, 0.15) * 0.02
        tube(stems, [o0, e], [0.002, 0.0015], sides=3)
        leaf(leaves, e, az, 0.1, 0.075, 0.042, serrated=True, droop=0.35, steps=4)
        for lat in (-1, 1):
            leaf(leaves, o0 + vec(az, 0.15) * 0.01, az + lat * 0.9, 0.05, 0.05, 0.03,
                 serrated=True, droop=0.35, steps=4)
    for k in range(1, 5):                               # opposite leaf pairs up the stem
        s_ = 1.0 - 0.13 * k
        for sg in (0, 1):
            az = base + k * 1.5708 + sg * math.pi
            o = np.array(pts[k])
            pitch = 0.45 - 0.05 * k
            e = o + vec(az, pitch) * 0.02 * s_
            tube(stems, [o, e], [0.0016, 0.0012], sides=3)
            leaf(leaves, e, az, pitch * 0.6, 0.07 * s_, 0.04 * s_, serrated=True, droop=0.25, steps=4)
            for lat in (-1, 1):
                leaf(leaves, o + vec(az, pitch) * 0.01 * s_, az + lat * 0.95, pitch * 0.5,
                     0.05 * s_, 0.03 * s_, serrated=True, droop=0.25, steps=4)

    def head(c, scale=1.0):
        for _ in range(16):
            d = vec(rnd.uniform(0, 6.28), rnd.uniform(-0.2, 1.5))
            tube(seeds, [c, c + d * rnd.uniform(0.014, 0.022) * scale], [0.001, 0.0003], sides=3)

    head(np.array(pts[-1]) + [0, 0, 0.004])
    for k, az in ((3, base), (4, base + 3.0)):
        s0 = np.array(pts[k])
        e = s0 + vec(az, 0.9) * 0.035
        tube(stems, [s0, e], [0.0016, 0.001], sides=3)
        head(e, 0.8)
    return stems, leaves, seeds


def make_plant(name, builder, seed, desc, leaf_cols, stem_cols, scale=1.0):
    d = MODELS / name
    tex = d / 'materials' / 'textures'
    leaf_texture(tex / 'leaf.png', *leaf_cols, seed=seed)
    streak_texture(tex / 'stem.png', seed + 5, stem_cols[0], stem_cols[1], n=128, stretch=4)
    stems, leaves, seeds = builder(seed)
    stems.write(d / 'meshes' / 'stems.obj', scale)
    leaves.write(d / 'meshes' / 'leaves.obj', scale)
    body = visual('stems', f'{name}/meshes/stems.obj', f'{name}/materials/textures/stem.png')
    body += visual('leaves', f'{name}/meshes/leaves.obj', f'{name}/materials/textures/leaf.png', rough=0.7)
    if seeds:
        streak_texture(tex / 'seed.png', seed + 9, (40, 28, 20), (86, 62, 40), n=128, stretch=4)
        seeds.write(d / 'meshes' / 'seeds.obj', scale)
        body += visual('seeds', f'{name}/meshes/seeds.obj', f'{name}/materials/textures/seed.png')
    write_sdf(d, name, body)
    write_config(d, name, desc)


def main():
    make_soil()
    make_hedge()
    make_fence()
    make_tree('tree_oak', 'oak', 1)
    make_tree('tree_acacia', 'acacia', 2)
    # Colours are chosen to stay inside the weed_detector HSV windows:
    #   crops -> blue-green (OpenCV hue 70-100), weeds -> yellow-green (hue 35-64)
    gn = ((28, 118, 98), (14, 72, 66), (110, 175, 160))            # leaf base, edge, vein
    gn_stem = ((30, 95, 85), (60, 125, 110))
    bj = ((100, 155, 38), (62, 112, 28), (170, 200, 95))
    bj_stem = ((80, 120, 40), (120, 150, 60))
    for i, nm in enumerate(['groundnuts', 'groundnuts_2', 'groundnuts_3']):
        make_plant(nm, groundnut_meshes, 100 + i, 'Groundnut (peanut) crop plant, visual only',
                   gn, gn_stem, scale=GROUNDNUT_SCALE)
    for i, nm in enumerate(['blackjack', 'blackjack_2', 'blackjack_3']):
        make_plant(nm, blackjack_meshes, 200 + i, 'Blackjack (Bidens pilosa) weed, visual only',
                   bj, bj_stem)
    names = sorted(p.name for p in MODELS.iterdir() if (p / 'model.sdf').exists())
    print('Models ready in', MODELS)
    for n in names:
        tris = sum(sum(1 for ln in open(o) if ln.startswith('f ')) for o in (MODELS / n).rglob('*.obj'))
        print(f'  {n:14s} {tris:6d} triangles')


if __name__ == '__main__':
    main()
