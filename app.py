import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cv2
import tempfile
import os
import time
import math
import heapq

st.set_page_config(
    page_title="3D Vision Field Mission Lab",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

FPS = 20
VIDEO_W, VIDEO_H = 1280, 720
MAP_W = 760
VIEW_W = VIDEO_W - MAP_W

ROOM = {
    "x_min": -3.8,
    "x_max": 3.8,
    "y_min": -2.8,
    "y_max": 2.8,
    "z_min": 0.0,
    "z_max": 2.3,
}

FURNITURE = [
    {
        "name": "Table",
        "center": np.array([0.0, 0.0, 0.38], dtype=float),
        "size": np.array([1.0, 0.6, 0.05], dtype=float),
    },
    {
        "name": "Workbench",
        "center": np.array([2.3, -1.3, 0.45], dtype=float),
        "size": np.array([1.2, 0.55, 0.05], dtype=float),
    },
    {
        "name": "Shelf",
        "center": np.array([-2.4, 1.4, 0.90], dtype=float),
        "size": np.array([0.35, 1.0, 1.6], dtype=float),
    },
    {
        "name": "Door",
        "center": np.array([-3.15, 0.0, 1.0], dtype=float),
        "size": np.array([0.1, 0.9, 2.0], dtype=float),
    },
    {
        "name": "Box",
        "center": np.array([1.4, 0.8, 0.18], dtype=float),
        "size": np.array([0.35, 0.25, 0.18], dtype=float),
    },
    {
        "name": "Cup",
        "center": np.array([0.18, 0.12, 0.47], dtype=float),
        "size": np.array([0.08, 0.08, 0.10], dtype=float),
    },
]

CUP_CENTER = next(item["center"] for item in FURNITURE if item["name"] == "Cup").copy()

BOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

JOINT_NAMES = [
    "pelvis",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
]

BONES = [
    (0, 1), (0, 2), (1, 3), (3, 5),
    (2, 4), (4, 6), (0, 7), (0, 8),
    (7, 9), (9, 11), (8, 10), (10, 12),
    (1, 13), (2, 13),
]

# ------------------------------------------------------------------------------
# Small geometry / utility helpers
# ------------------------------------------------------------------------------


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def box_corners(center, size):
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    dx, dy, dz = size / 2.0
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append([center[0] + sx * dx, center[1] + sy * dy, center[2] + sz * dz])
    return np.array(pts, dtype=float)


def sample_box_surface(center, size, n, rng):
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    pts = []
    for _ in range(int(n)):
        axis = int(rng.integers(0, 3))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        p = center.copy()
        p[axis] += sign * size[axis] / 2.0
        for a in range(3):
            if a != axis:
                p[a] += rng.uniform(-size[a] / 2.0, size[a] / 2.0)
        pts.append(p)
    return np.array(pts, dtype=float)


def rotation_from_yaw(yaw):
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    right = np.array([-s, c, 0.0], dtype=float)
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    forward = np.array([c, s, 0.0], dtype=float)
    return np.vstack([right, down, forward])


def project_points(points, cam_pose, K, image_size):
    points = np.asarray(points, dtype=float)
    C = np.asarray(cam_pose[:3], dtype=float)
    R = rotation_from_yaw(cam_pose[3])

    v = points - C
    pc = v @ R.T

    z = pc[:, 2]
    valid = z > 0.05

    u = np.full(len(points), -1.0, dtype=float)
    vv = np.full(len(points), -1.0, dtype=float)

    if np.any(valid):
        u[valid] = K[0, 0] * pc[valid, 0] / z[valid] + K[0, 2]
        vv[valid] = K[1, 1] * pc[valid, 1] / z[valid] + K[1, 2]

    inside = valid & (u >= 0) & (u < image_size[0]) & (vv >= 0) & (vv < image_size[1])
    return np.column_stack([u, vv]), inside, z


def back_project(u, v, depth, cam_pose, K):
    ray_cam = np.array(
        [
            (float(u) - K[0, 2]) * float(depth) / K[0, 0],
            (float(v) - K[1, 2]) * float(depth) / K[1, 1],
            float(depth),
        ],
        dtype=float,
    )
    R = rotation_from_yaw(cam_pose[3])
    return np.asarray(cam_pose[:3], dtype=float) + R.T @ ray_cam


def world_to_map_px(x, y, width, height, margin=55):
    xr = ROOM["x_max"] - ROOM["x_min"]
    yr = ROOM["y_max"] - ROOM["y_min"]
    scale = min((width - 2 * margin) / xr, (height - 2 * margin) / yr)
    px = int(width / 2.0 + float(x) * scale)
    py = int(height / 2.0 - float(y) * scale)
    return px, py


def draw_camera_marker(img, pose, width, height, color):
    px, py = world_to_map_px(pose[0], pose[1], width, height)
    yaw = float(pose[3])

    fx = math.cos(yaw)
    fy = -math.sin(yaw)

    rx = -fy
    ry = fx

    tip = (int(px + fx * 16), int(py + fy * 16))
    b1 = (int(px - fx * 6 + rx * 7), int(py - fy * 6 + ry * 7))
    b2 = (int(px - fx * 6 - rx * 7), int(py - fy * 6 - ry * 7))

    cv2.fillPoly(img, [np.array([tip, b1, b2], dtype=np.int32)], color)
    cv2.circle(img, (px, py), 3, (255, 255, 255), -1)


def mark_disk(grid, mask, pos, res, origin, radius):
    ny, nx = grid.shape
    x0, y0 = origin

    i0 = int(np.floor((pos[1] - radius - y0) / res))
    i1 = int(np.ceil((pos[1] + radius - y0) / res))
    j0 = int(np.floor((pos[0] - radius - x0) / res))
    j1 = int(np.ceil((pos[0] + radius - x0) / res))

    for i in range(max(0, i0), min(ny, i1 + 1)):
        y = y0 + (i + 0.5) * res
        for j in range(max(0, j0), min(nx, j1 + 1)):
            x = x0 + (j + 0.5) * res
            if (x - pos[0]) ** 2 + (y - pos[1]) ** 2 <= radius ** 2:
                grid[i, j] += 1.0
                if mask is not None:
                    mask[i, j] = True


def rotate_points_z(points, center, angle):
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    c, s = math.cos(float(angle)), math.sin(float(angle))
    v = points - center
    out = np.empty_like(v)
    out[:, 0] = c * v[:, 0] - s * v[:, 1]
    out[:, 1] = s * v[:, 0] + c * v[:, 1]
    out[:, 2] = v[:, 2]
    return out + center


def compute_scene_graph(person_pos, cup_pos, cam_pose):
    nodes = []
    for item in FURNITURE:
        pos = item["center"].copy()
        if item["name"] == "Cup":
            pos = np.asarray(cup_pos, dtype=float)
        nodes.append({"name": item["name"], "pos": pos})

    nodes.append({"name": "Person", "pos": np.asarray(person_pos, dtype=float)})

    def get_node(name):
        return next((n for n in nodes if n["name"] == name), None)

    edges = []

    table = get_node("Table")
    cup = get_node("Cup")
    door = get_node("Door")
    person = get_node("Person")
    shelf = get_node("Shelf")
    workbench = get_node("Workbench")

    if table is not None and cup is not None:
        dxy = float(np.linalg.norm(cup["pos"][:2] - table["pos"][:2]))
        dz = float(cup["pos"][2] - table["pos"][2])
        if 0.03 < dz < 0.35 and dxy < 0.45:
            conf = float(np.clip(1.0 - dxy / 0.45, 0.55, 0.99))
            edges.append(["Cup", "on", "Table", conf])

    if person is not None and door is not None:
        d = float(np.linalg.norm(person["pos"] - door["pos"]))
        if d < 1.6:
            edges.append(["Person", "near", "Door", float(np.clip(1.0 - d / 1.6, 0.45, 0.99))])

    if person is not None and table is not None:
        dxy = float(np.linalg.norm(person["pos"][:2] - table["pos"][:2]))
        if dxy < 1.4:
            edges.append(["Person", "near", "Table", float(np.clip(1.0 - dxy / 1.4, 0.45, 0.99))])
        if person["pos"][0] < table["pos"][0] - 0.35:
            edges.append(["Person", "left_of", "Table", 0.72])
        elif person["pos"][0] > table["pos"][0] + 0.35:
            edges.append(["Person", "right_of", "Table", 0.72])

    if person is not None and workbench is not None:
        dxy = float(np.linalg.norm(person["pos"][:2] - workbench["pos"][:2]))
        if dxy < 1.5:
            edges.append(["Person", "near", "Workbench", float(np.clip(1.0 - dxy / 1.5, 0.45, 0.99))])

    if person is not None and shelf is not None:
        dxy = float(np.linalg.norm(person["pos"][:2] - shelf["pos"][:2]))
        if dxy < 1.3:
            edges.append(["Person", "near", "Shelf", float(np.clip(1.0 - dxy / 1.3, 0.45, 0.99))])

    if cam_pose is not None:
        cam_pose = np.asarray(cam_pose, dtype=float)
        C = cam_pose[:3]
        yaw = float(cam_pose[3]) if cam_pose.size > 3 else 0.0

        # Important fix:
        # Use a 2D forward vector and compare it with the 2D relative vector.
        # Previously a 3D forward vector was dotted with v[:2], causing the ValueError.
        fwd_xy = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)

        for n in nodes:
            v = n["pos"] - C
            d = float(np.linalg.norm(v[:2]))
            if d < 3.0 and float(np.dot(fwd_xy, v[:2])) > 0.55 * d:
                edges.append(
                    [
                        n["name"],
                        "in_front_of",
                        "Camera",
                        float(np.clip(1.0 - d / 3.0, 0.35, 0.95)),
                    ]
                )

    df = (
        pd.DataFrame(edges, columns=["source", "relation", "target", "confidence"])
        if edges
        else pd.DataFrame(columns=["source", "relation", "target", "confidence"])
    )
    return nodes, df


def generate_human_pose(person_track, noise, rng):
    frames = len(person_track)

    base = np.array(
        [
            [0.00,  0.00,  0.00],  # pelvis
            [0.02,  0.18,  0.45],  # left_shoulder
            [0.02, -0.18,  0.45],  # right_shoulder
            [0.12,  0.28,  0.18],  # left_elbow
            [0.12, -0.28,  0.18],  # right_elbow
            [0.22,  0.32, -0.05],  # left_wrist
            [0.22, -0.32, -0.05],  # right_wrist
            [0.00,  0.09,  0.00],  # left_hip
            [0.00, -0.09,  0.00],  # right_hip
            [0.12,  0.10, -0.42],  # left_knee
            [0.12, -0.10, -0.42],  # right_knee
            [0.18,  0.10, -0.85],  # left_ankle
            [0.18, -0.10, -0.85],  # right_ankle
            [0.00,  0.00,  0.62],  # head
        ],
        dtype=float,
    )

    joints = np.zeros((frames, len(JOINT_NAMES), 3), dtype=float)
    phase = np.linspace(0.0, 6.0 * np.pi, frames)

    for f in range(frames):
        sw = 0.18 * math.sin(phase[f])
        loc = base.copy()

        loc[3, 0] += sw
        loc[5, 0] += 1.4 * sw
        loc[5, 2] += 0.08 * max(0.0, math.sin(phase[f]))

        loc[4, 0] -= sw
        loc[6, 0] -= 1.4 * sw
        loc[6, 2] += 0.08 * max(0.0, -math.sin(phase[f]))

        loc[9, 0] += 0.12 * max(0.0, math.sin(phase[f]))
        loc[11, 0] += 0.18 * max(0.0, math.sin(phase[f]))

        loc[10, 0] += 0.12 * max(0.0, -math.sin(phase[f]))
        loc[12, 0] += 0.18 * max(0.0, -math.sin(phase[f]))

        reach = max(0.0, math.sin(phase[f] * 0.45))
        loc[6, 0] += 0.35 * reach
        loc[6, 2] += 0.25 * reach

        if f == 0:
            vel = person_track[1] - person_track[0] if frames > 1 else np.array([1.0, 0.0, 0.0])
        else:
            vel = person_track[f] - person_track[f - 1]

        yaw = math.atan2(float(vel[1]), float(vel[0]))
        c, s = math.cos(yaw), math.sin(yaw)

        world_xy = np.column_stack([c * loc[:, 0] - s * loc[:, 1], s * loc[:, 0] + c * loc[:, 1]])

        joints[f, :, 0] = person_track[f, 0] + world_xy[:, 0]
        joints[f, :, 1] = person_track[f, 1] + world_xy[:, 1]
        joints[f, :, 2] = person_track[f, 2] + loc[:, 2]

    joints += rng.normal(scale=0.003 + noise * 0.02, size=joints.shape)

    window = 5
    smoothed = np.zeros_like(joints)
    half = window // 2

    for j in range(len(JOINT_NAMES)):
        for d in range(3):
            sig = np.convolve(joints[:, j, d], np.ones(window) / window, mode="same")
            sig[:half] = joints[:half, j, d]
            sig[-half:] = joints[-half:, j, d]
            smoothed[:, j, d] = sig

    pelvis = smoothed[:, 0:1, :]
    dist_from_pelvis = np.linalg.norm(smoothed - pelvis, axis=2)
    uncertainty = 0.018 + noise * 0.55 + 0.01 * dist_from_pelvis

    wrists = smoothed[:, [5, 6], :]
    reach_distances = np.linalg.norm(wrists - pelvis, axis=2)
    max_reach = float(np.max(reach_distances)) if reach_distances.size else 0.65

    return smoothed, uncertainty, max_reach


def simulate_docking(noise, seed, tol=0.12):
    rng = np.random.default_rng(int(seed) + 99)

    target = np.array([2.45, -0.75, 0.0], dtype=float)
    start = np.array([-2.60, 1.90, 0.0], dtype=float)

    pose = start.copy()
    heading = 0.0
    path = [pose.copy()]

    max_steps = 260
    dt = 0.05

    for _ in range(max_steps):
        err = target[:2] - pose[:2]
        dist = float(np.linalg.norm(err))
        if dist < tol:
            break

        desired = math.atan2(float(err[1]), float(err[0]))
        da = wrap_angle(desired - heading)
        heading += float(np.clip(da, -0.18, 0.18))

        if abs(da) < 0.55:
            step = 0.035 * max(0.25, 1.0 - dist / 6.0)
            pose[:2] += step * np.array([math.cos(heading), math.sin(heading)])

        pose[:2] += rng.normal(scale=0.002 + noise * 0.08, size=2)
        pose[2] = 0.0
        path.append(pose.copy())

    path = np.array(path, dtype=float)
    final_dist = float(np.linalg.norm(target[:2] - path[-1, :2]))

    return {
        "path": path,
        "target": target,
        "start": start,
        "tol": float(tol),
        "success": bool(final_dist < tol),
        "final_error": final_dist,
        "time_s": float(len(path) * dt),
    }


def estimate_dimensions(noise, seed):
    rng = np.random.default_rng(int(seed) + 7)

    true_center = np.array([1.75, 0.45, 0.25], dtype=float)
    true_size = np.array([0.42, 0.28, 0.19], dtype=float)
    true_yaw = 0.35

    pts = sample_box_surface(true_center, true_size, 320, rng)
    pts = rotate_points_z(pts, true_center, true_yaw)
    pts += rng.normal(scale=0.003 + noise * 0.05, size=pts.shape)

    est_center = pts.mean(axis=0)
    centered = pts - est_center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)

    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]

    proj = centered @ vecs
    extents = proj.max(axis=0) - proj.min(axis=0)
    est_dims = np.sort(extents)[::-1]
    true_dims = np.sort(true_size)[::-1]

    err_cm = np.abs(est_dims - true_dims) * 100.0

    return {
        "points": pts,
        "center": est_center,
        "axes": vecs,
        "dims": est_dims,
        "true_dims": true_dims,
        "true_center": true_center,
        "true_yaw": true_yaw,
        "error_cm": err_cm,
        "mean_error_cm": float(np.mean(err_cm)),
    }


def is_free_cell(grid, cell):
    ny, nx = grid.shape
    i, j = cell
    return 0 <= i < ny and 0 <= j < nx and float(grid[i, j]) < 0.5


def nearest_free_cell(grid, cell):
    if is_free_cell(grid, cell):
        return cell

    ny, nx = grid.shape
    max_r = max(ny, nx)

    for r in range(1, max_r + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                c = (cell[0] + di, cell[1] + dj)
                if is_free_cell(grid, c):
                    return c

    return cell


def astar(grid, start, goal):
    start = nearest_free_cell(grid, start)
    goal = nearest_free_cell(grid, goal)

    if start == goal:
        return [start]

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), start))

    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (current[0] + dr, current[1] + dc)
                if not is_free_cell(grid, nxt):
                    continue

                tentative_g = g_score[current] + math.hypot(dr, dc)
                if tentative_g < g_score.get(nxt, 1e18):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f_score[nxt] = tentative_g + heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f_score[nxt], nxt))

    return []


def world_to_cell(pos, res, origin, shape):
    ny, nx = shape
    x, y = float(pos[0]), float(pos[1])
    j = int(np.floor((x - origin[0]) / res))
    i = int(np.floor((y - origin[1]) / res))
    return (np.clip(i, 0, ny - 1), np.clip(j, 0, nx - 1))


def cell_to_world(i, j, res, origin):
    x = origin[0] + (j + 0.5) * res
    y = origin[1] + (i + 0.5) * res
    return np.array([x, y], dtype=float)


def plan_nav_path(grid, res, origin, start_world, goal_world):
    start = world_to_cell(start_world, res, origin, grid.shape)
    goal = world_to_cell(goal_world, res, origin, grid.shape)
    cells = astar(grid, start, goal)

    if not cells:
        return np.array([np.asarray(start_world)[:2], np.asarray(goal_world)[:2]], dtype=float)

    return np.array([cell_to_world(i, j, res, origin) for i, j in cells], dtype=float)


def generate_forecast(seed, noise, past=25, future=15):
    rng = np.random.default_rng(int(seed) + 31)
    t = np.linspace(0.0, 2.0 * np.pi, past + future)

    obj0 = np.column_stack(
        [
            2.2 * np.cos(t * 0.55) - 0.2,
            1.4 * np.sin(t * 0.55),
            0.25 + 0.02 * np.sin(t * 2.0),
        ]
    )

    obj1 = np.column_stack(
        [
            np.linspace(-2.0, 2.0, past + future),
            -1.0 + 0.45 * np.sin(t * 1.5),
            0.28 + 0.01 * np.cos(t * 3.0),
        ]
    )

    tracks = [obj0, obj1]
    past_tracks = []
    future_gt = []
    preds = []
    sigmas = []
    ades = []
    fdes = []

    for tr in tracks:
        p = tr[:past]
        gt = tr[past:]

        vel = np.mean(np.diff(p, axis=0), axis=0)
        last = p[-1]

        k = np.arange(1, future + 1, dtype=float)[:, None]
        pred = last + vel * k

        sigma = 0.025 + 0.012 * k + float(noise) * 0.2 * k
        sigma = sigma.squeeze()

        ade = float(np.mean(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)))
        fde = float(np.linalg.norm(pred[-1, :2] - gt[-1, :2]))

        past_tracks.append(p)
        future_gt.append(gt)
        preds.append(pred)
        sigmas.append(sigma)
        ades.append(ade)
        fdes.append(fde)

    return {
        "past": past_tracks,
        "future_gt": future_gt,
        "pred": preds,
        "sigma": sigmas,
        "ade": ades,
        "fde": fdes,
    }


def calibration_residuals(points, cam_pose, tx, ty, tz, rot):
    K = np.array(
        [
            [550.0, 0.0, 320.0],
            [0.0, 550.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    points = np.asarray(points, dtype=float)
    cam_pose = np.asarray(cam_pose, dtype=float)

    pix0, inside0, _ = project_points(points, cam_pose, K, (640, 480))

    C = cam_pose[:3].copy()
    yaw = float(cam_pose[3])
    R = rotation_from_yaw(yaw)

    C2 = C + R.T @ np.array([tx, ty, tz], dtype=float)
    yaw2 = yaw + rot
    pert_pose = np.array([C2[0], C2[1], C2[2], yaw2], dtype=float)

    pix1, inside1, _ = project_points(points, pert_pose, K, (640, 480))

    vis = inside0 & inside1
    if np.count_nonzero(vis) < 5:
        mag = math.sqrt(tx * tx + ty * ty + tz * tz) + abs(rot) * 0.5
        return np.full(len(points), 1.5 + 90.0 * mag, dtype=float)

    res = np.linalg.norm(pix0[vis] - pix1[vis], axis=1)
    return res


# ------------------------------------------------------------------------------
# Core synthetic demo data builder
# ------------------------------------------------------------------------------


def _build_demo_data(seed, frames, map_points, dynamic_objects, noise, motion_scale):
    t_start = time.perf_counter()

    seed = int(seed)
    frames = max(10, int(frames))
    map_points = max(50, int(map_points))
    dynamic_objects = max(0, int(dynamic_objects))
    noise = float(noise)
    motion_scale = float(motion_scale)

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------------------------
    # Static 3D map points: walls, furniture surfaces, floor features
    # --------------------------------------------------------------------------
    pts = []

    wall_n = int(map_points * 0.42)
    for _ in range(wall_n):
        side = int(rng.integers(0, 4))
        if side == 0:
            p = [ROOM["x_min"], rng.uniform(ROOM["y_min"], ROOM["y_max"]), rng.uniform(ROOM["z_min"], ROOM["z_max"])]
        elif side == 1:
            p = [ROOM["x_max"], rng.uniform(ROOM["y_min"], ROOM["y_max"]), rng.uniform(ROOM["z_min"], ROOM["z_max"])]
        elif side == 2:
            p = [rng.uniform(ROOM["x_min"], ROOM["x_max"]), ROOM["y_min"], rng.uniform(ROOM["z_min"], ROOM["z_max"])]
        else:
            p = [rng.uniform(ROOM["x_min"], ROOM["x_max"]), ROOM["y_max"], rng.uniform(ROOM["z_min"], ROOM["z_max"])]
        pts.append(p)

    furn_each = max(6, int(map_points * 0.38 / max(1, len(FURNITURE))))
    for item in FURNITURE:
        surf = sample_box_surface(item["center"], item["size"], furn_each, rng)
        pts.extend(surf.tolist())

    while len(pts) < map_points:
        pts.append(
            [
                rng.uniform(ROOM["x_min"] + 0.2, ROOM["x_max"] - 0.2),
                rng.uniform(ROOM["y_min"] + 0.2, ROOM["y_max"] - 0.2),
                0.02,
            ]
        )

    static_points = np.array(pts[:map_points], dtype=float)

    # --------------------------------------------------------------------------
    # Ground-truth camera trajectory and noisy visual odometry estimate
    # --------------------------------------------------------------------------
    tt = np.linspace(0.0, 2.0 * np.pi, frames)

    x = (2.0 + 0.35 * np.sin(3.0 * tt)) * motion_scale * np.cos(tt)
    y = (1.25 + 0.25 * np.cos(2.0 * tt)) * motion_scale * np.sin(tt)
    z = 1.25 + 0.08 * np.sin(5.0 * tt)

    x = np.clip(x, ROOM["x_min"] + 0.35, ROOM["x_max"] - 0.35)
    y = np.clip(y, ROOM["y_min"] + 0.35, ROOM["y_max"] - 0.35)

    vel = np.gradient(np.column_stack([x, y]), axis=0)
    yaw = np.arctan2(vel[:, 1], vel[:, 0])
    gt_pose = np.column_stack([x, y, z, yaw])

    est_pose = np.zeros_like(gt_pose)
    est_pose[0] = gt_pose[0] + np.concatenate(
        [
            rng.normal(scale=noise * 0.25, size=3),
            [rng.normal(scale=noise * 0.03)],
        ]
    )

    lost = rng.random(frames) < (noise * 0.8)
    lost[0] = False

    for i in range(1, frames):
        delta = gt_pose[i] - gt_pose[i - 1]
        if lost[i]:
            est_pose[i] = est_pose[i - 1]
        else:
            est_pose[i] = est_pose[i - 1] + delta + np.concatenate(
                [
                    rng.normal(scale=noise * 0.18, size=3),
                    [rng.normal(scale=noise * 0.02)],
                ]
            )

    ate = np.linalg.norm(est_pose[:, :2] - gt_pose[:, :2], axis=1)
    ate_rmse = float(np.sqrt(np.mean(ate ** 2)))
    tracking_loss = float(100.0 * np.mean(lost))

    # --------------------------------------------------------------------------
    # Dynamic object tracks
    # --------------------------------------------------------------------------
    dynamic_tracks = np.zeros((dynamic_objects, frames, 3), dtype=float)

    if dynamic_objects > 0:
        dynamic_tracks[0, :, 0] = np.linspace(-2.6, 2.6, frames)
        dynamic_tracks[0, :, 1] = 0.9 + 0.35 * np.sin(np.linspace(0.0, 4.0 * np.pi, frames))
        dynamic_tracks[0, :, 2] = 0.22

    if dynamic_objects > 1:
        ang = np.linspace(0.0, 2.0 * np.pi, frames)
        dynamic_tracks[1, :, 0] = 1.7 + 0.65 * np.cos(ang)
        dynamic_tracks[1, :, 1] = -1.1 + 0.55 * np.sin(ang)
        dynamic_tracks[1, :, 2] = 0.28

    if dynamic_objects > 2:
        dynamic_tracks[2, :, 0] = -1.2 + 0.2 * np.cos(np.linspace(0.0, 6.0 * np.pi, frames))
        dynamic_tracks[2, :, 1] = -1.7 + np.linspace(0.0, 2.8, frames)
        dynamic_tracks[2, :, 2] = 0.25

    if dynamic_objects > 3:
        dynamic_tracks[3, :, 0] = 2.8 - np.linspace(0.0, 4.5, frames)
        dynamic_tracks[3, :, 1] = 1.8 - np.linspace(0.0, 3.2, frames)
        dynamic_tracks[3, :, 2] = 0.30

    # --------------------------------------------------------------------------
    # Person track used for scene graph and human pose
    # --------------------------------------------------------------------------
    person_track = np.zeros((frames, 3), dtype=float)
    person_track[:, 0] = np.linspace(-1.8, 1.9, frames)
    person_track[:, 1] = 1.35 - 2.0 * np.linspace(0.0, 1.0, frames) + 0.18 * np.sin(np.linspace(0.0, 5.0 * np.pi, frames))
    person_track[:, 2] = 0.95

    # --------------------------------------------------------------------------
    # RGB-D object localization demo
    # --------------------------------------------------------------------------
    img_w, img_h = 640, 480
    K = np.array(
        [
            [550.0, 0.0, img_w / 2.0],
            [0.0, 550.0, img_h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    box_item = next(item for item in FURNITURE if item["name"] == "Box")
    box_center = box_item["center"].copy()
    box_size = box_item["size"].copy()

    rgb_frame = 0
    best = None

    for f in range(frames):
        pix, inside, zc = project_points(box_center.reshape(1, 3), est_pose[f], K, (img_w, img_h))
        if inside[0] and zc[0] > 0.6:
            rgb_frame = f
            best = (pix[0].copy(), float(zc[0]))
            break

    if best is None:
        pix, inside, zc = project_points(box_center.reshape(1, 3), est_pose[0], K, (img_w, img_h))
        center_pix = pix[0].copy()
        center_depth = float(max(0.8, zc[0] if zc[0] > 0 else 0.8))
    else:
        center_pix, center_depth = best

    corners = box_corners(box_center, box_size)
    cpix, cinside, cz = project_points(corners, est_pose[rgb_frame], K, (img_w, img_h))

    rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    cv2.rectangle(rgb, (0, 0), (img_w, int(img_h * 0.55)), (55, 75, 95), -1)
    cv2.rectangle(rgb, (0, int(img_h * 0.55)), (img_w, img_h), (45, 50, 55), -1)

    for gx in range(0, img_w, 40):
        cv2.line(rgb, (gx, img_h // 2), (gx - 100, img_h), (70, 80, 90), 1)

    depth_map = np.full((img_h, img_w), 4.0, dtype=float)

    visible = cpix[cinside].astype(np.int32)
    if len(visible) >= 3:
        hull = cv2.convexHull(visible)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        depth_map[mask > 0] = center_depth
        cv2.fillPoly(rgb, [hull], (90, 170, 220))

    for a, b in BOX_EDGES:
        if cinside[a] and cinside[b]:
            pt1 = tuple(cpix[a].astype(int))
            pt2 = tuple(cpix[b].astype(int))
            cv2.line(rgb, pt1, pt2, (255, 255, 255), 2)

    if 0 <= center_pix[0] < img_w and 0 <= center_pix[1] < img_h:
        cv2.putText(
            rgb,
            "Box",
            (int(center_pix[0]) + 10, int(center_pix[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    dmin, dmax = 0.4, 4.0
    dn = np.clip((depth_map - dmin) / (dmax - dmin), 0.0, 1.0)
    dn8 = (dn * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(dn8, cv2.COLORMAP_TURBO)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

    depth_noise_val = float(abs(math.sin(float(seed) * 0.7)) * noise * 0.8 + 0.004)
    depth_meas = float(center_depth + depth_noise_val)

    if center_pix[0] < 0 or center_pix[1] < 0:
        p_est = box_center + np.array([noise * 0.2, noise * 0.15, noise * 0.1], dtype=float)
    else:
        p_est = back_project(center_pix[0], center_pix[1], depth_meas, est_pose[rgb_frame], K)

    loc_rmse = float(np.linalg.norm(p_est - box_center))

    rgb_data = {
        "rgb": rgb,
        "depth": depth_color,
        "K": K,
        "cam_pose": est_pose[rgb_frame].copy(),
        "center_pix": center_pix.copy(),
        "depth_true": float(center_depth),
        "depth_meas": depth_meas,
        "box_center": box_center.copy(),
        "box_size": box_size.copy(),
        "loc_rmse": loc_rmse,
        "frame": int(rgb_frame),
    }

    # --------------------------------------------------------------------------
    # Default scene graph
    # --------------------------------------------------------------------------
    graph_nodes, graph_edges = compute_scene_graph(person_track[0], CUP_CENTER, est_pose[0])

    # --------------------------------------------------------------------------
    # Dynamic obstacle occupancy mapping
    # --------------------------------------------------------------------------
    res = 0.12
    nx = int((ROOM["x_max"] - ROOM["x_min"]) / res) + 1
    ny = int((ROOM["y_max"] - ROOM["y_min"]) / res) + 1
    origin = (ROOM["x_min"], ROOM["y_min"])

    static_grid = np.zeros((ny, nx), dtype=float)
    for p in static_points:
        if p[2] < 1.9:
            j = int((p[0] - origin[0]) / res)
            i = int((p[1] - origin[1]) / res)
            if 0 <= i < ny and 0 <= j < nx:
                static_grid[i, j] = 1.0

    dyn_obs = np.zeros((ny, nx), dtype=float)
    true_dyn = np.zeros((ny, nx), dtype=bool)

    for d in range(dynamic_objects):
        for f in range(frames):
            mark_disk(dyn_obs, true_dyn, dynamic_tracks[d, f], res, origin, radius=0.22)

    dynamic_prob = dyn_obs / float(max(1, frames))
    classified_grid = ((dynamic_prob > 0.03) & (static_grid < 0.5)).astype(float)

    true_dyn_count = int(np.count_nonzero(true_dyn & (static_grid < 0.5)))
    detected = classified_grid > 0.5
    tp = int(np.count_nonzero(detected & true_dyn & (static_grid < 0.5)))
    fp = int(np.count_nonzero(detected & ~(true_dyn & (static_grid < 0.5))))

    obstacle_recall = float(tp / max(1, true_dyn_count))
    false_dynamic_rate = float(fp / max(1, np.count_nonzero(detected)))
    update_delay = float(max(1.0, 1.5 + noise * 25.0))

    occupancy = {
        "static_grid": static_grid,
        "dynamic_obs": dyn_obs,
        "classified_grid": classified_grid,
        "res": res,
        "origin": origin,
        "recall": obstacle_recall,
        "false_dynamic_rate": false_dynamic_rate,
        "update_delay": update_delay,
    }

    # --------------------------------------------------------------------------
    # Human pose and reach
    # --------------------------------------------------------------------------
    joints, uncertainty, max_reach = generate_human_pose(person_track, noise, rng)

    pose_data = {
        "joints": joints,
        "uncertainty": uncertainty,
        "max_reach": max_reach,
        "mpjpe_cm": float(100.0 * (noise * 0.8 + 0.015)),
        "reach_f1": float(max(0.65, 0.96 - noise * 3.0)),
    }

    # --------------------------------------------------------------------------
    # Visual docking
    # --------------------------------------------------------------------------
    docking = simulate_docking(noise=noise, seed=seed, tol=0.12)

    # --------------------------------------------------------------------------
    # 3D object dimension estimation
    # --------------------------------------------------------------------------
    dimensions = estimate_dimensions(noise=noise, seed=seed)

    # --------------------------------------------------------------------------
    # Indoor semantic navigation
    # --------------------------------------------------------------------------
    nav_places = {
        "Workbench": np.array([2.3, -1.3], dtype=float),
        "Table": np.array([0.0, 0.0], dtype=float),
        "Door": np.array([-3.15, 0.0], dtype=float),
        "Shelf": np.array([-2.4, 1.4], dtype=float),
    }

    nav_start = np.array([-2.6, 1.8], dtype=float)
    nav_goal_name = "Workbench"
    nav_path = plan_nav_path(static_grid, res, origin, nav_start, nav_places[nav_goal_name])

    nav = {
        "places": nav_places,
        "start": nav_start,
        "goal": nav_goal_name,
        "path": nav_path,
    }

    # --------------------------------------------------------------------------
    # Trajectory forecasting
    # --------------------------------------------------------------------------
    forecast = generate_forecast(seed=seed, noise=noise, past=25, future=15)

    # --------------------------------------------------------------------------
    # Cross-sensor calibration monitor
    # --------------------------------------------------------------------------
    calib_points = static_points[:: max(1, len(static_points) // 120)][:120]
    nominal_pose = gt_pose[0].copy()
    default_pert = np.array([0.025, -0.018, 0.012, 0.008], dtype=float)

    res0 = calibration_residuals(
        calib_points,
        nominal_pose,
        default_pert[0],
        default_pert[1],
        default_pert[2],
        default_pert[3],
    )
    mean_res0 = float(np.mean(res0)) if len(res0) else 0.0

    calibration = {
        "points": calib_points,
        "pose": nominal_pose,
        "default_pert": default_pert,
        "mean_residual": mean_res0,
        "threshold": 3.0,
    }

    runtime_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "seed": seed,
        "frames": frames,
        "map_points": map_points,
        "dynamic_objects": dynamic_objects,
        "noise": noise,
        "motion_scale": motion_scale,
        "static_points": static_points,
        "furniture": FURNITURE,
        "gt_pose": gt_pose,
        "est_pose": est_pose,
        "ate_per_frame": ate,
        "ate_rmse": ate_rmse,
        "tracking_loss": tracking_loss,
        "runtime_ms": runtime_ms,
        "dynamic_tracks": dynamic_tracks,
        "person_track": person_track,
        "rgb": rgb_data,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "occupancy": occupancy,
        "pose": pose_data,
        "docking": docking,
        "dimensions": dimensions,
        "nav": nav,
        "forecast": forecast,
        "calibration": calibration,
    }


@st.cache_data(show_spinner=False)
def get_demo_data(seed, frames, map_points, dynamic_objects, noise, motion_scale):
    return _build_demo_data(seed, frames, map_points, dynamic_objects, noise, motion_scale)


# ------------------------------------------------------------------------------
# Live video renderer
# ------------------------------------------------------------------------------


def make_video_context(data):
    static = data["static_points"]
    idx = np.linspace(0, len(static) - 1, min(220, len(static)), dtype=int)
    draw_pts = static[idx]

    K_view = np.array(
        [
            [520.0, 0.0, VIEW_W / 2.0],
            [0.0, 520.0, VIDEO_H / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return draw_pts, K_view


def render_demo_frame(data, f, draw_pts, K_view):
    img = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)

    # Left: top-down map
    cv2.rectangle(img, (0, 0), (MAP_W - 1, VIDEO_H - 1), (16, 20, 28), -1)

    x1, y1 = world_to_map_px(ROOM["x_min"], ROOM["y_min"], MAP_W, VIDEO_H)
    x2, y2 = world_to_map_px(ROOM["x_max"], ROOM["y_max"], MAP_W, VIDEO_H)
    cv2.rectangle(
        img,
        (min(x1, x2), min(y1, y2)),
        (max(x1, x2), max(y1, y2)),
        (70, 85, 100),
        2,
    )

    for item in data["furniture"]:
        cx, cy = float(item["center"][0]), float(item["center"][1])
        sx, sy = float(item["size"][0]) / 2.0, float(item["size"][1]) / 2.0
        coords = [(cx - sx, cy - sy), (cx + sx, cy - sy), (cx + sx, cy + sy), (cx - sx, cy + sy)]
        pxs = [world_to_map_px(x, y, MAP_W, VIDEO_H) for x, y in coords]
        cv2.polylines(img, [np.array(pxs, dtype=np.int32).reshape(-1, 1, 2)], True, (90, 110, 130), 1)

    for p in draw_pts:
        px, py = world_to_map_px(p[0], p[1], MAP_W, VIDEO_H)
        intensity = int(np.clip(120 - 30 * p[2], 50, 180))
        cv2.circle(img, (px, py), 1, (intensity, intensity, intensity), -1)

    if f > 0:
        gt_pts = np.array(
            [world_to_map_px(x, y, MAP_W, VIDEO_H) for x, y in data["gt_pose"][: f + 1, :2]],
            dtype=np.int32,
        )
        est_pts = np.array(
            [world_to_map_px(x, y, MAP_W, VIDEO_H) for x, y in data["est_pose"][: f + 1, :2]],
            dtype=np.int32,
        )
        cv2.polylines(img, [gt_pts.reshape(-1, 1, 2)], False, (220, 160, 60), 2)
        cv2.polylines(img, [est_pts.reshape(-1, 1, 2)], False, (60, 220, 120), 2)

    draw_camera_marker(img, data["est_pose"][f], MAP_W, VIDEO_H, (40, 90, 255))

    tracks = data["dynamic_tracks"]
    for d in range(tracks.shape[0]):
        tr = tracks[d, : f + 1]
        if len(tr) > 1:
            tpts = np.array([world_to_map_px(x, y, MAP_W, VIDEO_H) for x, y in tr[:, :2]], dtype=np.int32)
            cv2.polylines(img, [tpts.reshape(-1, 1, 2)], False, (40, 180, 255), 2)

        px, py = world_to_map_px(tr[-1, 0], tr[-1, 1], MAP_W, VIDEO_H)
        cv2.circle(img, (px, py), 5, (40, 180, 255), -1)

    # Right: synthetic monocular camera view
    view = np.zeros((VIDEO_H, VIEW_W, 3), dtype=np.uint8)
    cv2.rectangle(view, (0, 0), (VIEW_W - 1, int(VIDEO_H * 0.55)), (45, 60, 75), -1)
    cv2.rectangle(view, (0, int(VIDEO_H * 0.55)), (VIEW_W - 1, VIDEO_H - 1), (35, 40, 48), -1)

    pix, inside, zc = project_points(data["static_points"], data["est_pose"][f], K_view, (VIEW_W, VIDEO_H))
    vis_idx = np.where(inside)[0]
    if len(vis_idx) > 0:
        vis_idx = vis_idx[np.argsort(-zc[vis_idx])]
        for idx in vis_idx[:260]:
            u, v = pix[idx]
            depth = zc[idx]
            bright = int(np.clip(255 - depth * 35, 40, 230))
            cv2.circle(view, (int(u), int(v)), 2, (bright, bright, min(255, bright + 30)), -1)

    for d in range(tracks.shape[0]):
        p = tracks[d, f]
        pixd, insided, _ = project_points(p.reshape(1, 3), data["est_pose"][f], K_view, (VIEW_W, VIDEO_H))
        if insided[0]:
            cv2.circle(view, (int(pixd[0, 0]), int(pixd[0, 1])), 6, (40, 180, 255), -1)

    box = next((item for item in data["furniture"] if item["name"] == "Box"), None)
    if box is not None:
        corners = box_corners(box["center"], box["size"])
        bpix, bins, _ = project_points(corners, data["est_pose"][f], K_view, (VIEW_W, VIDEO_H))
        for a, b in BOX_EDGES:
            if bins[a] and bins[b]:
                cv2.line(
                    view,
                    tuple(bpix[a].astype(int)),
                    tuple(bpix[b].astype(int)),
                    (90, 220, 255),
                    2,
                )

    img[:, MAP_W:VIDEO_W] = view
    cv2.line(img, (MAP_W, 0), (MAP_W, VIDEO_H), (200, 200, 200), 2)

    cv2.putText(
        img,
        "MONOCULAR VO + DYNAMIC MAPPING AUTO DEMO",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        img,
        f"frame {f + 1}/{data['frames']}  ATE {data['ate_rmse']:.3f} m  loss {data['tracking_loss']:.1f}%",
        (20, VIDEO_H - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )

    cv2.putText(
        img,
        "top-down map",
        (20, VIDEO_H - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
    )

    cv2.putText(
        img,
        "synthetic camera view",
        (MAP_W + 20, VIDEO_H - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
    )

    return img


def generate_demo_video(seed, frames, map_points, dynamic_objects, noise, motion_scale):
    data = get_demo_data(seed, frames, map_points, dynamic_objects, noise, motion_scale)
    draw_pts, K_view = make_video_context(data)

    tmp_dir = tempfile.gettempdir()
    basename = f"mission3d_{seed}_{frames}_{map_points}_{dynamic_objects}_{int(noise * 1000)}_{int(motion_scale * 100)}"

    out = None
    out_path = None

    # Prefer MP4-compatible codecs so Streamlit can play the generated file directly.
    for codec, suffix in [("avc1", ".mp4"), ("mp4v", ".mp4")]:
        candidate = os.path.join(tmp_dir, basename + suffix)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(candidate, fourcc, FPS, (VIDEO_W, VIDEO_H))
        if writer.isOpened():
            out = writer
            out_path = candidate
            break

    if out is None or out_path is None:
        return None

    for f in range(frames):
        frame = render_demo_frame(data, f, draw_pts, K_view)
        out.write(frame)

    out.release()

    try:
        with open(out_path, "rb") as fh:
            video_bytes = fh.read()
        os.remove(out_path)
        return video_bytes
    except Exception:
        try:
            os.remove(out_path)
        except Exception:
            pass
        return None


@st.cache_data(show_spinner=False)
def get_demo_video(seed, frames, map_points, dynamic_objects, noise, motion_scale):
    return generate_demo_video(seed, frames, map_points, dynamic_objects, noise, motion_scale)


# ------------------------------------------------------------------------------
# Plotly figure helpers
# ------------------------------------------------------------------------------


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig_trajectory(data):
    fig = go.Figure()

    pts = data["static_points"][:: max(1, len(data["static_points"]) // 350)]

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.8, color="lightgray"),
            name="Sparse map",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=data["gt_pose"][:, 0],
            y=data["gt_pose"][:, 1],
            z=data["gt_pose"][:, 2],
            mode="lines",
            line=dict(color="orange", width=6),
            name="Ground truth",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=data["est_pose"][:, 0],
            y=data["est_pose"][:, 1],
            z=data["est_pose"][:, 2],
            mode="lines+markers",
            line=dict(color="red", width=4),
            marker=dict(size=2),
            name="Estimated VO",
        )
    )

    fig.update_layout(
        title="Monocular Visual Odometry and Sparse Map",
        height=560,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1.0, z=0.45),
        ),
    )
    return fig


def fig_error(data):
    df = pd.DataFrame({"frame": np.arange(1, len(data["ate_per_frame"]) + 1), "error_m": data["ate_per_frame"]})
    fig = go.Figure(
        go.Scatter(
            x=df["frame"],
            y=df["error_m"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="crimson"),
            fillcolor="rgba(220, 50, 50, 0.15)",
        )
    )
    fig.update_layout(
        title="Trajectory Error Over Time",
        xaxis_title="Frame",
        yaxis_title="XY error (m)",
        height=320,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def fig_scene_graph(nodes, edges, cam_pose):
    fig = go.Figure()

    node_map = {n["name"]: n["pos"] for n in nodes}
    if cam_pose is not None:
        node_map["Camera"] = np.asarray(cam_pose[:3], dtype=float)

    colors = {
        "Table": "#4C78A8",
        "Workbench": "#F58518",
        "Shelf": "#54A24B",
        "Door": "#E45756",
        "Box": "#B279A2",
        "Cup": "#FF9DA6",
        "Person": "#72B7B2",
        "Camera": "#444444",
    }

    for name, pos in node_map.items():
        fig.add_trace(
            go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode="markers+text",
                marker=dict(size=8, color=colors.get(name, "#888888")),
                text=[name],
                textposition="top center",
                name=name,
                showlegend=False,
            )
        )

    for _, row in edges.iterrows():
        src = row["source"]
        tgt = row["target"]
        if src in node_map and tgt in node_map:
            a = node_map[src]
            b = node_map[tgt]
            fig.add_trace(
                go.Scatter3d(
                    x=[a[0], b[0], None],
                    y=[a[1], b[1], None],
                    z=[a[2], b[2], None],
                    mode="lines",
                    line=dict(color="rgba(100,100,100,0.65)", width=3),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title="Semantic Scene Graph",
        height=560,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="manual",
            aspectratio=dict(x=1.3, y=1.0, z=0.55),
        ),
    )
    return fig


def grid_xy(grid, res, origin):
    ny, nx = grid.shape
    x = origin[0] + (np.arange(nx) + 0.5) * res
    y = origin[1] + (np.arange(ny) + 0.5) * res
    return x, y


def fig_heatmap(grid, res, origin, title, colorscale="Greys"):
    x, y = grid_xy(grid, res, origin)
    fig = go.Figure(
        go.Heatmap(
            z=grid,
            x=x,
            y=y,
            colorscale=colorscale,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        height=420,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def fig_dynamic_map(occ, tracks, frame):
    x, y = grid_xy(occ["classified_grid"], occ["res"], occ["origin"])

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=occ["static_grid"],
            x=x,
            y=y,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(120,120,120,0.45)"]],
            showscale=False,
            hoverongaps=False,
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=occ["classified_grid"],
            x=x,
            y=y,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,127,14,0.75)"]],
            showscale=False,
            hoverongaps=False,
        )
    )

    for d in range(tracks.shape[0]):
        tr = tracks[d, : frame + 1]
        fig.add_trace(
            go.Scatter(
                x=tr[:, 0],
                y=tr[:, 1],
                mode="lines+markers",
                name=f"Dynamic {d + 1}",
                line=dict(width=3),
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        title="Dynamic Obstacle Map Update",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        height=520,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def fig_pose(joints, frame, uncertainty, max_reach):
    fig = go.Figure()

    pts = joints[frame]

    for a, b in BONES:
        fig.add_trace(
            go.Scatter3d(
                x=[pts[a, 0], pts[b, 0], None],
                y=[pts[a, 1], pts[b, 1], None],
                z=[pts[a, 2], pts[b, 2], None],
                mode="lines",
                line=dict(color="#2ca02c", width=5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    sizes = 6 + 35 * uncertainty[frame]

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers+text",
            marker=dict(size=sizes, color=uncertainty[frame], colorscale="Viridis", showscale=True),
            text=JOINT_NAMES,
            textposition="top center",
            name="Joints",
        )
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    pelvis = pts[0]
    fig.add_trace(
        go.Scatter3d(
            x=pelvis[0] + max_reach * np.cos(theta),
            y=pelvis[1] + max_reach * np.sin(theta),
            z=np.full_like(theta, pelvis[2]),
            mode="lines",
            line=dict(color="rgba(255,165,0,0.65)", width=4),
            name="Reach envelope",
        )
    )

    fig.update_layout(
        title="3D Human Pose and Approximate Reach Envelope",
        height=580,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=1.2),
        ),
    )
    return fig


def fig_docking(dock):
    fig = go.Figure()

    path = dock["path"]

    fig.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines+markers",
            name="Robot path",
            line=dict(width=4),
            marker=dict(size=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[dock["start"][0]],
            y=[dock["start"][1]],
            mode="markers+text",
            name="Start",
            marker=dict(size=14, symbol="triangle-up"),
            text=["Start"],
            textposition="top center",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[dock["target"][0]],
            y=[dock["target"][1]],
            mode="markers+text",
            name="Marker",
            marker=dict(size=16, symbol="x", color="red"),
            text=["Station"],
            textposition="top center",
        )
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    fig.add_trace(
        go.Scatter(
            x=dock["target"][0] + dock["tol"] * np.cos(theta),
            y=dock["target"][1] + dock["tol"] * np.sin(theta),
            mode="lines",
            name="Tolerance",
            line=dict(dash="dot", color="gray"),
        )
    )

    fig.update_layout(
        title="Visual Docking and Landmark Alignment",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        yaxis_scaleanchor="x",
        height=520,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def obb_corners(center, dims, axes):
    half = np.asarray(dims, dtype=float) / 2.0
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=float)
                pts.append(center + axes @ local)
    return np.array(pts, dtype=float)


def fig_dimensions(dim):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=dim["points"][:, 0],
            y=dim["points"][:, 1],
            z=dim["points"][:, 2],
            mode="markers",
            marker=dict(size=2, color="lightgray"),
            name="Point cloud",
        )
    )

    est_corners = obb_corners(dim["center"], dim["dims"], dim["axes"])
    for a, b in BOX_EDGES:
        fig.add_trace(
            go.Scatter3d(
                x=[est_corners[a, 0], est_corners[b, 0], None],
                y=[est_corners[a, 1], est_corners[b, 1], None],
                z=[est_corners[a, 2], est_corners[b, 2], None],
                mode="lines",
                line=dict(color="cyan", width=5),
                hoverinfo="skip",
                showlegend=(a == 0 and b == 1),
                name="Estimated OBB",
            )
        )

    true_corners = rotate_points_z(box_corners(dim["true_center"], dim["true_dims"]), dim["true_center"], dim["true_yaw"])
    for a, b in BOX_EDGES:
        fig.add_trace(
            go.Scatter3d(
                x=[true_corners[a, 0], true_corners[b, 0], None],
                y=[true_corners[a, 1], true_corners[b, 1], None],
                z=[true_corners[a, 2], true_corners[b, 2], None],
                mode="lines",
                line=dict(color="orange", width=3, dash="dot"),
                hoverinfo="skip",
                showlegend=(a == 0 and b == 1),
                name="True box",
            )
        )

    fig.update_layout(
        title="3D Object Dimension Estimator",
        height=560,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
    )
    return fig


def fig_nav(grid, res, origin, path, places, start, goal_name):
    x, y = grid_xy(grid, res, origin)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=x,
            y=y,
            colorscale=[[0, "rgba(240,240,240,1)"], [1, "rgba(80,80,80,1)"]],
            showscale=False,
            hoverongaps=False,
        )
    )

    if path is not None and len(path) > 1:
        fig.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines+markers",
                name="Planned path",
                line=dict(color="#d62728", width=4),
                marker=dict(size=4),
            )
        )

    for name, pos in places.items():
        fig.add_trace(
            go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode="markers+text",
                name=name,
                marker=dict(size=12, symbol="square"),
                text=[name],
                textposition="top center",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[start[0]],
            y=[start[1]],
            mode="markers+text",
            name="Robot",
            marker=dict(size=15, symbol="triangle-up", color="#1f77b4"),
            text=["Robot"],
            textposition="bottom center",
        )
    )

    fig.update_layout(
        title=f"Indoor Navigation Semantic Map — Goal: {goal_name}",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        yaxis_scaleanchor="x",
        height=540,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def uncertainty_band(xy, sigma):
    xy = np.asarray(xy, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    tangents = np.gradient(xy, axis=0)
    norms = np.linalg.norm(tangents, axis=1) + 1e-6

    normals = np.column_stack([-tangents[:, 1] / norms, tangents[:, 0] / norms])

    upper = xy + normals * sigma[:, None]
    lower = xy - normals * sigma[:, None]
    return upper, lower


def fig_forecast(fc):
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for i in range(len(fc["past"])):
        color = colors[i % len(colors)]

        past = fc["past"][i]
        gt = fc["future_gt"][i]
        pred = fc["pred"][i]
        sigma = fc["sigma"][i]

        fig.add_trace(
            go.Scatter(
                x=past[:, 0],
                y=past[:, 1],
                mode="lines",
                name=f"Track {i + 1} past",
                line=dict(color=color, width=5),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=gt[:, 0],
                y=gt[:, 1],
                mode="lines",
                name=f"Track {i + 1} truth",
                line=dict(color=color, dash="dot", width=3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pred[:, 0],
                y=pred[:, 1],
                mode="lines+markers",
                name=f"Track {i + 1} forecast",
                line=dict(color=color, dash="dash", width=3),
                marker=dict(size=4),
            )
        )

        upper, lower = uncertainty_band(pred[:, :2], sigma)
        fig.add_trace(
            go.Scatter(
                x=upper[:, 0],
                y=upper[:, 1],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lower[:, 0],
                y=lower[:, 1],
                mode="lines",
                fill="tonexty",
                fillcolor=hex_to_rgba(color, 0.16),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Short-Horizon 3D Trajectory Forecasting with Uncertainty",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        yaxis_scaleanchor="x",
        height=540,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def fig_calibration(values, threshold):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        values = np.array([0.0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=values,
            mode="lines+markers",
            name="Mean reprojection residual",
            line=dict(color="#d62728", width=3),
            marker=dict(size=4),
        )
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="green", annotation_text="Alert threshold")

    fig.update_layout(
        title="Cross-Sensor Calibration Drift Monitor",
        xaxis_title="Time step",
        yaxis_title="Residual (px)",
        height=380,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


# ------------------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------------------

st.sidebar.title("🎛️ Mission Controls")
st.sidebar.caption("All data is synthetic and generated locally. No API key required.")

seed = st.sidebar.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
duration = st.sidebar.slider("Demo duration (s)", min_value=2, max_value=8, value=4, step=1)
map_points = st.sidebar.slider("Sparse map points", min_value=80, max_value=600, value=260, step=20)
dynamic_objects = st.sidebar.slider("Dynamic objects", min_value=0, max_value=4, value=2, step=1)
noise = st.sidebar.slider("Sensor / process noise", min_value=0.0, max_value=0.08, value=0.02, step=0.005)
motion_scale = st.sidebar.slider("Traversal scale", min_value=0.6, max_value=1.2, value=1.0, step=0.05)

frames = int(duration * FPS)

# ------------------------------------------------------------------------------
# Build data
# ------------------------------------------------------------------------------

data = get_demo_data(
    seed=int(seed),
    frames=int(frames),
    map_points=int(map_points),
    dynamic_objects=int(dynamic_objects),
    noise=float(noise),
    motion_scale=float(motion_scale),
)

# ------------------------------------------------------------------------------
# Header and auto preview
# ------------------------------------------------------------------------------

st.title("🛰️ 3D Vision Field Mission Lab")
st.caption(
    "Auto-loaded synthetic field demo: monocular motion, RGB-D localization, scene graphs, "
    "dynamic mapping, pose, docking, dimensions, semantic navigation, forecasting, and calibration monitoring."
)

preview_key = f"{seed}|{frames}|{map_points}|{dynamic_objects}|{noise}|{motion_scale}"

if st.session_state.get("preview_key") != preview_key:
    preview_holder = st.empty()
    draw_pts_preview, K_view_preview = make_video_context(data)
    limit = min(frames, 30)

    for ff in range(limit):
        frame_bgr = render_demo_frame(data, ff, draw_pts_preview, K_view_preview)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        preview_holder.image(frame_rgb, use_container_width=True)
        time.sleep(0.045)

    preview_holder.empty()
    st.session_state.preview_key = preview_key

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Trajectory ATE RMSE", f"{data['ate_rmse']:.3f} m")
m2.metric("Tracking loss", f"{data['tracking_loss']:.1f} %")
m3.metric("Map points", f"{len(data['static_points'])}")
m4.metric("Dynamic objects", f"{data['dynamic_objects']}")
m5.metric("Build runtime", f"{data['runtime_ms']:.1f} ms")

st.subheader("🎥 Generated live demo video")

with st.spinner("Generating MP4 demo video..."):
    video_bytes = get_demo_video(
        seed=int(seed),
        frames=int(frames),
        map_points=int(map_points),
        dynamic_objects=int(dynamic_objects),
        noise=float(noise),
        motion_scale=float(motion_scale),
    )

if video_bytes:
    st.video(video_bytes, format="video/mp4", loop=True, muted=True)
else:
    st.warning("Video writer could not open an MP4 codec on this system. The auto preview above still ran.")

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------

tab_vo, tab_rgbd, tab_graph, tab_dynamic, tab_pose, tab_dock, tab_dim, tab_nav, tab_forecast, tab_calib = st.tabs(
    [
        "🧭 Visual Odometry",
        "📦 RGB-D Localization",
        "🕸️ Scene Graph",
        "🚧 Dynamic Mapping",
        "🧍 Human Pose",
        "🔌 Visual Docking",
        "📐 Dimensions",
        "🗺️ Semantic Nav",
        "🔮 Forecasting",
        "🎯 Calibration",
    ]
)

# ----- Visual Odometry -----
with tab_vo:
    st.plotly_chart(fig_trajectory(data), use_container_width=True)
    st.plotly_chart(fig_error(data), use_container_width=True)
    st.markdown(
        """
        **What this module shows**
        - Sparse 3D map points from synthetic room geometry
        - Ground-truth camera path
        - Noisy monocular visual odometry estimate
        - Trajectory error over time
        """
    )

# ----- RGB-D Localization -----
with tab_rgbd:
    c1, c2 = st.columns(2)
    with c1:
        st.image(data["rgb"]["rgb"], caption="Synthetic RGB view", use_container_width=True)
    with c2:
        st.image(data["rgb"]["depth"], caption="Aligned depth view", use_container_width=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Localization RMSE", f"{data['rgb']['loc_rmse']:.3f} m")
    r2.metric("True depth", f"{data['rgb']['depth_true']:.3f} m")
    r3.metric("Measured depth", f"{data['rgb']['depth_meas']:.3f} m")
    r4.metric("RGB-D frame", f"{data['rgb']['frame']}")

    st.subheader("Interactive 3D reprojection")
    extra_depth_noise = st.slider("Extra depth noise (m)", 0.0, 0.10, 0.01, step=0.005)

    deterministic_jitter = abs(math.sin(float(seed) * 1.7))
    test_depth = data["rgb"]["depth_true"] + extra_depth_noise * deterministic_jitter

    if data["rgb"]["center_pix"][0] >= 0 and data["rgb"]["center_pix"][1] >= 0:
        test_point = back_project(
            data["rgb"]["center_pix"][0],
            data["rgb"]["center_pix"][1],
            test_depth,
            data["rgb"]["cam_pose"],
            data["rgb"]["K"],
        )
    else:
        test_point = data["rgb"]["box_center"] + np.array([0.01, 0.01, 0.01])

    cmp = pd.DataFrame(
        {
            "component": ["X", "Y", "Z"],
            "ground_truth_m": data["rgb"]["box_center"],
            "estimated_m": test_point,
            "error_m": np.abs(test_point - data["rgb"]["box_center"]),
        }
    )
    st.dataframe(cmp, use_container_width=True)

# ----- Scene Graph -----
with tab_graph:
    graph_frame = st.slider("Scene graph temporal frame", 0, frames - 1, frames // 2, key="graph_frame")

    cup_pos = CUP_CENTER + np.array(
        [
            0.03 * math.sin(graph_frame * 0.25),
            0.02 * math.cos(graph_frame * 0.20),
            0.0,
        ],
        dtype=float,
    )

    nodes, edges = compute_scene_graph(
        data["person_track"][graph_frame],
        cup_pos,
        data["est_pose"][graph_frame],
    )

    st.dataframe(edges, use_container_width=True)
    st.plotly_chart(fig_scene_graph(nodes, edges, data["est_pose"][graph_frame]), use_container_width=True)

# ----- Dynamic Mapping -----
with tab_dynamic:
    occ = data["occupancy"]

    d1, d2, d3 = st.columns(3)
    d1.metric("Obstacle recall", f"{occ['recall']:.2f}")
    d2.metric("False dynamic rate", f"{occ['false_dynamic_rate']:.2f}")
    d3.metric("Update delay", f"{occ['update_delay']:.1f} frames")

    dyn_frame = st.slider("Map update frame", 0, frames - 1, frames - 1, key="dyn_frame")
    st.plotly_chart(fig_dynamic_map(occ, data["dynamic_tracks"], dyn_frame), use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.plotly_chart(
            fig_heatmap(occ["static_grid"], occ["res"], occ["origin"], "Static structure grid", "Greys"),
            use_container_width=True,
        )
    with cc2:
        st.plotly_chart(
            fig_heatmap(occ["classified_grid"], occ["res"], occ["origin"], "Dynamic class grid", "Oranges"),
            use_container_width=True,
        )

# ----- Human Pose -----
with tab_pose:
    pose_frame = st.slider("Pose frame", 0, frames - 1, frames // 2, key="pose_frame")

    joints = data["pose"]["joints"]
    uncertainty = data["pose"]["uncertainty"]

    p1, p2, p3 = st.columns(3)
    p1.metric("Simulated MPJPE", f"{data['pose']['mpjpe_cm']:.1f} cm")
    p2.metric("Reach classification F1", f"{data['pose']['reach_f1']:.2f}")
    p3.metric("Max estimated reach", f"{data['pose']['max_reach']:.2f} m")

    st.plotly_chart(fig_pose(joints, pose_frame, uncertainty, data["pose"]["max_reach"]), use_container_width=True)

    pose_df = pd.DataFrame(
        {
            "joint": JOINT_NAMES,
            "x_m": joints[pose_frame, :, 0],
            "y_m": joints[pose_frame, :, 1],
            "z_m": joints[pose_frame, :, 2],
            "uncertainty_m": uncertainty[pose_frame],
        }
    )
    st.dataframe(pose_df, use_container_width=True)

# ----- Visual Docking -----
with tab_dock:
    dock_noise = st.slider("Docking controller noise", 0.0, 0.08, float(noise), step=0.005, key="dock_noise")
    dock_tol = st.slider("Docking tolerance (m)", 0.05, 0.30, 0.12, step=0.01, key="dock_tol")

    dock = simulate_docking(noise=dock_noise, seed=seed, tol=dock_tol)

    q1, q2, q3 = st.columns(3)
    q1.metric("Docking success", "YES" if dock["success"] else "NO")
    q2.metric("Final position error", f"{dock['final_error']:.3f} m")
    q3.metric("Time to dock", f"{dock['time_s']:.2f} s")

    st.plotly_chart(fig_docking(dock), use_container_width=True)

# ----- Dimensions -----
with tab_dim:
    dim_noise = st.slider("Dimension point noise", 0.0, 0.08, float(noise), step=0.005, key="dim_noise")
    dim = estimate_dimensions(noise=dim_noise, seed=seed)

    z1, z2, z3 = st.columns(3)
    z1.metric("Mean dimension error", f"{dim['mean_error_cm']:.2f} cm")
    z2.metric("Estimated length", f"{dim['dims'][0]:.3f} m")
    z3.metric("Estimated width", f"{dim['dims'][1]:.3f} m")

    st.plotly_chart(fig_dimensions(dim), use_container_width=True)

    dim_df = pd.DataFrame(
        {
            "dimension": ["Length", "Width", "Height"],
            "true_m": dim["true_dims"],
            "estimated_m": dim["dims"],
            "error_cm": dim["error_cm"],
        }
    )
    st.dataframe(dim_df, use_container_width=True)

# ----- Semantic Navigation -----
with tab_nav:
    occ = data["occupancy"]
    places = data["nav"]["places"]

    goal_name = st.selectbox("Goal place", list(places.keys()), index=0, key="nav_goal")
    start_name = st.selectbox("Start place", ["Robot", "Door", "Table", "Shelf", "Workbench"], index=0, key="nav_start")

    if start_name == "Robot":
        start_pos = data["nav"]["start"]
    else:
        start_pos = places[start_name]

    goal_pos = places[goal_name]
    path = plan_nav_path(occ["static_grid"], occ["res"], occ["origin"], start_pos, goal_pos)

    st.plotly_chart(
        fig_nav(occ["static_grid"], occ["res"], occ["origin"], path, places, start_pos, goal_name),
        use_container_width=True,
    )

    if len(path) > 1:
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        st.metric("Planned path length", f"{length:.2f} m")
    else:
        st.warning("No collision-free path found for the selected start and goal.")

# ----- Forecasting -----
with tab_forecast:
    fc = data["forecast"]

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Track 1 ADE", f"{fc['ade'][0]:.3f} m")
    f2.metric("Track 1 FDE", f"{fc['fde'][0]:.3f} m")
    f3.metric("Track 2 ADE", f"{fc['ade'][1]:.3f} m")
    f4.metric("Track 2 FDE", f"{fc['fde'][1]:.3f} m")

    st.plotly_chart(fig_forecast(fc), use_container_width=True)

# ----- Calibration Monitor -----
with tab_calib:
    cal = data["calibration"]

    tx = st.slider("Camera/depth X offset (m)", -0.08, 0.08, float(cal["default_pert"][0]), step=0.005, key="cal_tx")
    ty = st.slider("Camera/depth Y offset (m)", -0.08, 0.08, float(cal["default_pert"][1]), step=0.005, key="cal_ty")
    tz = st.slider("Camera/depth Z offset (m)", -0.08, 0.08, float(cal["default_pert"][2]), step=0.005, key="cal_tz")
    rot = st.slider("Yaw offset (rad)", -0.05, 0.05, float(cal["default_pert"][3]), step=0.005, key="cal_rot")
    drift = st.slider("Drift rate", 0.0, 0.002, 0.0002, step=0.0001, key="cal_drift")

    current_res = calibration_residuals(cal["points"], cal["pose"], tx, ty, tz, rot)
    current_mean = float(np.mean(current_res)) if len(current_res) else 0.0

    status = "DRIFT ALERT" if current_mean > cal["threshold"] else "OK"

    s1, s2, s3 = st.columns(3)
    s1.metric("Mean residual", f"{current_mean:.2f} px")
    s2.metric("Threshold", f"{cal['threshold']:.2f} px")
    s3.metric("Status", status)

    series = []
    max_series = min(frames, 50)

    for ff in range(max_series):
        scale = 1.0 + drift * ff
        r = calibration_residuals(cal["points"], data["gt_pose"][ff], tx * scale, ty * scale, tz * scale, rot * scale)
        series.append(float(np.mean(r)) if len(r) else 0.0)

    st.plotly_chart(fig_calibration(np.array(series), cal["threshold"]), use_container_width=True)