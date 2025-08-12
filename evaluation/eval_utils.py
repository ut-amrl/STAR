import os
import math
import json
import csv
from collections import defaultdict
import pandas as pd

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