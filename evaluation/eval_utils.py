import os
import math
import json
import csv
from collections import defaultdict
import random
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timezone

def load_task_metadata(
    task_file_path: str, 
    benchmark_dir: str, 
    task_types: list[str],
    prefix: str = "tasks",
    versions: list = ["", "_wp_only", "_recaption_wp_only"]
):
    # Initialize the dictionary with all required keys
    
    task_dict = defaultdict(list)
    for task_type in task_types:
        task_dict[task_type] = []

    with open(task_file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        category, filename = line.split("/", 1)
        if category in task_dict:
            full_path = os.path.join(benchmark_dir, prefix, category, filename)
            for version in versions:
                task_dict[f"{category}{version}"].append(full_path)

    return task_dict


def load_annotated_waypoints(benchmark_dir: str):
    waypoint_dir = os.path.join(benchmark_dir, "annotations", "annotated_waypoints")
    result = {}

    for fname in os.listdir(waypoint_dir):
        if fname.endswith(".json"):
            bagname = fname.replace(".json", "")
            fpath = os.path.join(waypoint_dir, fname)
            with open(fpath, "r") as f:
                raw_data = json.load(f)

            # Apply floor to start time, ceil to end time
            processed_data = {
                k: [math.floor(v[0]), math.ceil(v[1])]
                for k, v in raw_data.items()
            }
            result[bagname] = processed_data

    return result


def load_waypoints(benchmark_dir: str):
    waypoint_file = os.path.join(benchmark_dir, "annotations", "waypoints.txt")
    result = {}

    with open(waypoint_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != ["waypoint", "x", "y", "theta"]:
            raise ValueError(f"Unexpected header in waypoint file: {header}")

        for row in reader:
            if len(row) != 4:
                raise ValueError(f"Malformed line: {row}")
            wp_id = str(int(row[0]))  # Normalize "00" → "0"
            coords = list(map(float, row[1:]))
            result[wp_id] = coords

    return result

def load_data_metadata(data_dir: str):
    result = {}
    for bagname in os.listdir(data_dir):
        bag_path = os.path.join(data_dir, bagname)
        if not os.path.isdir(bag_path):
            continue  # skip non-directory entries

        caption_file = os.path.join(bag_path, "caption_gpt4o.json")
        if os.path.exists(caption_file):
            result[bagname] = caption_file
        else:
            raise FileNotFoundError(f"Missing caption_gpt4o.json in {bag_path}")
    
    return result

def load_virtualhome_data_metadata(data_dir: str, caption_type: str = "gt"):
    result = {}
    for dataname in os.listdir(data_dir):
        data_path = os.path.join(data_dir, dataname)
        if not os.path.isdir(data_path):
            continue  # skip non-directory entries
        simulation_data_dir = os.path.join(data_path, "0")

        caption_file = os.path.join(simulation_data_dir, f"caption_gpt4o_{caption_type}.json")
        if os.path.exists(caption_file):
            result[dataname] = caption_file
    return result

def load_virtualhome_annotations(data_dir: str):
    all_annotations = {}
    for dataname in os.listdir(data_dir):
        data_path = os.path.join(data_dir, dataname)
        if not os.path.isdir(data_path):
            continue  # skip non-directory entries
        simulation_data_dir = os.path.join(data_path, "0")
        
        all_annotations[dataname] = {}
        
        annotation_file = os.path.join(simulation_data_dir, "gt_annotations.json")
        try:
            with open(annotation_file, "r") as f:
                frame_annotations = json.load(f)
            if type(frame_annotations) != list:
                raise ValueError(f"Unexpected type of annotations: {type(frame_annotations)}")
        except:
            continue
        all_annotations[dataname]["frames"] = frame_annotations
        
        object_placement_file = os.path.join(simulation_data_dir, "object_placement.csv")
        object_placement = pd.read_csv(object_placement_file).to_dict(orient="records")
        all_annotations[dataname]["object_placements"] = object_placement
        
    return all_annotations

import rospy
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import (
    ChangeVirtualHomeGraphSrv,
    ChangeVirtualHomeGraphSrvRequest,
)
def set_virtulhome_scene(graph_path: str, scene_id: int = None) -> bool:
    """
    Set the virtual home scene by changing the graph.
    """
    rospy.wait_for_service("/moma/change_virtualhome_graph", timeout=10)
    try:
        change_graph_service = rospy.ServiceProxy("/moma/change_virtualhome_graph", ChangeVirtualHomeGraphSrv)
        request = ChangeVirtualHomeGraphSrvRequest()
        request.graph_path = graph_path
        if scene_id is not None:
            request.scene_id = scene_id
        response = change_graph_service(request)
        return response.success
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        return False
    
def load_virtualhome_poses(data_dir: str, run_dirs: list, ds: float = 0.25) -> list:
    """
    Collect all runs for a given scene_id, resample each by uniform arc-length,
    and return traversable (x,z) poses.

    Args:
        data_dir: Root directory containing subfolders.
        scene_id: e.g. 4 -> matches 'scene4_01', 'scene4_00_classonly', etc.
        ds: arc-length step for resampling.
        ensure_endpoints: keep first/last original points.

    Returns:
        List of ((x,z,0.0), 0.0) tuples across all runs.
    """
    runs = []
    for dataname in run_dirs:
        pose_path = os.path.join(data_dir, dataname, "0", f"pd_{dataname}.txt")
        if not os.path.isfile(pose_path):
            continue

        pts_xz = []
        with open(pose_path, "r") as f:
            for line in f.readlines()[1:]:
                values = line.strip().split()
                if len(values) < 4:
                    continue
                try:
                    x1, y1, z1 = map(float, values[1 + 5*3 : 4 + 5*3])
                    x2, y2, z2 = map(float, values[1 + 6*3 : 4 + 6*3])
                except Exception:
                    continue
                x, y, z = (x1+x2)/2.0, (y1+y2)/2.0, (z1+z2)/2.0
                pts_xz.append((x, z))
        if len(pts_xz) >= 2:
            runs.append(pts_xz)

    # resample each run
    # resampled_runs = resample_runs_by_arclength(runs, ds=ds)
    waypoints = resample_runs_by_arclength(runs, ds=ds, merge=True, rmin=ds)
    poses = [((float(x), float(z), 0.0), 0.0) for x,z in waypoints]
    
    # def debug_plot_poses(poses, outpath="debug.png"):
    #     import matplotlib.pyplot as plt
    #     xs = [p[0][0] for p in poses]
    #     zs = [p[0][1] for p in poses]
    #     plt.figure(figsize=(6,6))
    #     plt.scatter(xs, zs, s=5, c="blue")
    #     plt.axis("equal")
    #     plt.tight_layout()
    #     plt.savefig(outpath, dpi=150)
    #     plt.close()
    # debug_plot_poses(poses)
    
    return poses


def resample_runs_by_arclength(
    runs, ds=0.05, ensure_endpoints=True, *,
    merge=False, rmin=None
):
    """
    Resample each (x,y) trajectory to uniform arc-length spacing `ds`.

    Args:
        runs: list of iterables of (x,y)
        ds:   spacing along each trajectory
        ensure_endpoints: include first/last original point in each resampled run
        merge: if True, return ONE merged array of points across all runs
        rmin: if provided (and merge=True), thin merged points so neighbors are >= rmin

    Returns:
        If merge=False (default): list[np.ndarray(Mi,2)]
        If merge=True: np.ndarray(M,2)
    """
    
    def _grid_thin(points, rmin):
        """
        Fast, dependency-free thinning: keep at most one point per rmin-sized grid cell.
        Not as uniform as Poisson-disk, but simple and robust.
        """
        if len(points) == 0:
            return points
        cell = np.floor(points / float(rmin)).astype(np.int64)
        # unique rows, keep first occurrence
        _, keep_idx = np.unique(cell, axis=0, return_index=True)
        keep_idx.sort()
        return points[keep_idx]
    
    out = []
    for traj in runs:
        pts = np.asarray(traj, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            out.append(np.empty((0, 2)))
            continue
        m = np.isfinite(pts).all(axis=1)
        pts = pts[m]
        # drop consecutive duplicates
        if len(pts) >= 2:
            keep = np.ones(len(pts), dtype=bool)
            keep[1:] = np.any(np.diff(pts, axis=0) != 0.0, axis=1)
            pts = pts[keep]
        if len(pts) < 2:
            out.append(pts[:1])
            continue

        seg = np.diff(pts, axis=0)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:,0], seg[:,1]))])
        total = s[-1]
        if total == 0.0:
            out.append(pts[:1])
            continue

        s_new = np.arange(0.0, total + 1e-9, ds)
        if ensure_endpoints:
            if s_new[0] != 0.0:
                s_new = np.insert(s_new, 0, 0.0)
            if s_new[-1] != total:
                s_new = np.append(s_new, total)
        else:
            if total - s_new[-1] < 0.5 * ds:
                s_new[-1] = total

        x_new = np.interp(s_new, s, pts[:,0])
        y_new = np.interp(s_new, s, pts[:,1])
        out.append(np.column_stack([x_new, y_new]))

    if not merge:
        return out

    # Merge all runs into one array
    merged = np.vstack([p for p in out if len(p) > 0]) if out else np.empty((0,2))
    if rmin is not None and rmin > 0:
        merged = _grid_thin(merged, rmin=rmin)
    return merged

def load_virtualhom_memory_sg(data_dir: str, datanames: list[str], bag_unix_times: list) -> list:
    graphs = ""
    for dataname, unix_time in zip(datanames, bag_unix_times):
        path = os.path.join(data_dir, dataname, "0", f"graph.json")
        with open(path, "r") as f:
            graph = json.load(f)
        if graph is None:
            raise ValueError(f"Graph not found in {path}")
        graph["nodes"] = [{"id": n["id"],
                           "category": n["category"],
                           "class_name": n["class_name"],
                           "object_position": [n["obj_transform"]["position"][0], n["obj_transform"]["position"][1], n["obj_transform"]["position"][2]],
                           "properties": n["properties"],
                           "states": n["states"]} for n in graph["nodes"]]
        random.shuffle(graph["nodes"])
        graph_yaml = yaml.safe_dump(graph, sort_keys=False, allow_unicode=True)
        readable_time = datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        graph_str = f"This is how the environment was like at {readable_time}:\n{graph_yaml}\n\n"
        graphs += graph_str
    return graphs