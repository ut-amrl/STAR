import os
import math
import json
import csv
from collections import defaultdict

def load_task_metadata(task_file_path: str, benchmark_dir: str):
    # Initialize the dictionary with all required keys
    
    task_dict = defaultdict(list)
    task_dict["unambiguous"] = []
    task_dict["spatial"] = []
    task_dict["spatial_temporal"] = []

    with open(task_file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        category, filename = line.split("/", 1)
        if category in task_dict:
            full_path = os.path.join(benchmark_dir, "tasks", category, filename)
            task_dict[category].append(full_path)
            task_dict[f"{category}_wp_only"].append(full_path)

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
