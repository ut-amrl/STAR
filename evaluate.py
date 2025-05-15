import argparse
from datetime import datetime, timezone
from pathlib import Path
from pymilvus import utility
from collections import defaultdict
from tqdm import tqdm

from evaluation.eval_utils import *
from agent.agent import Agent
from memory.memory import MilvusMemory, MemoryItem
from agent.utils.memloader import remember

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the agent.")
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        required=True,
        help="Path to the directory containing the task files.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the data files.",
    )
    parser.add_argument(
        "--task_file",
        type=str,
        default="evaluation/config/tasks.txt",
        help="Path to the task file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation/outputs/results.json",
        help="Optional path to save evaluation results (default: evaluation/outputs/results.json)"
    )
    args = parser.parse_args()
    return args

def evaluate_one_retrieval_task(args, agent: Agent, task: dict):
    question = f"Today is {args.current_pretty_date}. {task['task']}"
    result = agent.run(question=question, graph_type="retrieval")
    start_t, end_t = task["mem_obs_time"]
    if result is None:
        return (False, (start_t, end_t, None))
    retrieval_t = result["timestamp"]
    return (start_t <= retrieval_t and retrieval_t <= end_t, (start_t, end_t, retrieval_t))

def evaluate(args):
    agent = Agent()
    
    data_metadata = load_data_metadata(args.data_dir)
    task_metadata = load_task_metadata(args.task_file, args.benchmark_dir)
    bag_waypoint_mapping = load_annotated_waypoints(args.benchmark_dir)
    waypoints = load_waypoints(args.benchmark_dir)
    
    output = {
        "task_metadata": task_metadata,
    }
    results = defaultdict(list)
    for task_type, task_paths in task_metadata.items():
        # if task_type != "spatial_temporal":
            # continue # TODO: remove this line to evaluate all tasks
        if task_type != "unambiguous" and task_type != "unambiguous_wp_only":
            continue # TODO: remove this line to evaluate all tasks
        
        # progress bar
        total_tasks = 0
        task_path_to_num_tasks = {}
        for task_path in task_paths:
            with open(task_path, "r") as f:
                task_data = json.load(f)
                n_tasks = len(task_data["tasks"])
                task_path_to_num_tasks[task_path] = n_tasks
                total_tasks += n_tasks
        pbar = tqdm(total=total_tasks, desc=f"Evaluating [{task_type}]", unit="task")
        
        for task_path in task_paths:
            task_id = Path(task_path).stem
            
            with open(task_path, "r") as f:
                task_data = json.load(f)
                
            db_name = f"cobot_{task_type}_{task_id}"
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            obs_savepath = f"data/cobot/{task_type}_{task_id}_{timestamp}/"
            memory = MilvusMemory(db_name, obs_savepth=obs_savepath)
            memory.reset()
                
            bag_unix_times = {}
            for bagname, (date_str, time_str) in task_data["bag_time_mapping"].items():
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                    if bagname == task_data["bagnames"][-1]:
                        try:
                            args.current_pretty_date = dt.strftime("%b %-d, %Y")  # Linux/Mac
                        except ValueError:
                            args.current_pretty_date = dt.strftime("%b %#d, %Y")  # Windows fallback
                    # dt = dt.replace(tzinfo=timezone.utc)
                    unix_time = dt.timestamp()
                    bag_unix_times[bagname] = unix_time
                except Exception as e:
                    raise ValueError(f"Failed to parse datetime for bag '{bagname}': {e}")
                
            inpaths, time_offsets, viddirs = [], [], []
            for bagname in task_data["bagnames"]:
                inpaths.append(data_metadata[bagname])
                time_offsets.append(bag_unix_times[bagname])
                viddirs.append(os.path.join(args.data_dir, bagname, "images"))
            if "wp_only" in task_type:
                remember(memory, inpaths, time_offsets, viddirs, waypoint_only=True)
            else:
                remember(memory, inpaths, time_offsets, viddirs, waypoint_only=False)
            agent.set_memory(memory)
                
            for task in task_data["tasks"]:
                bagname = task["bagname_current"]
                wp_id = task["instance_wp"]
                # Add current_obs_time
                if bagname in bag_waypoint_mapping and wp_id in bag_waypoint_mapping[bagname]:
                    rel_start, rel_end = bag_waypoint_mapping[bagname][wp_id]
                    base_time = bag_unix_times[bagname]
                    task["current_obs_time"] = [base_time + rel_start - 1, base_time + rel_end + 1]
                else:
                    raise ValueError(f"Missing waypoint '{wp_id}' in bag '{bagname}'")
                
                # Add current_obs_pose
                if wp_id in waypoints:
                    task["current_obs_pose"] = waypoints[wp_id]
                else:
                    raise ValueError(f"Missing waypoint '{wp_id}' in global waypoints")
                
                # Annotate memory observation time & pose (if available)
                if "bagname_in_mem" in task and "wp_in_mem" in task:
                    mem_bag = task["bagname_in_mem"]
                    mem_wp = task["wp_in_mem"]

                    if mem_bag in bag_waypoint_mapping and mem_wp in bag_waypoint_mapping[mem_bag]:
                        rel_start, rel_end = bag_waypoint_mapping[mem_bag][mem_wp]
                        base_time = bag_unix_times[mem_bag]
                        task["mem_obs_time"] = [base_time + rel_start - 1, base_time + rel_end + 1]
                    else:
                        raise ValueError(f"Missing memory waypoint '{mem_wp}' in bag '{mem_bag}'")

                    if mem_wp in waypoints:
                        task["mem_obs_pose"] = waypoints[mem_wp]
                    else:
                        raise ValueError(f"Missing memory waypoint '{mem_wp}' in global waypoints")
                else:
                    task["mem_obs_time"] = task["current_obs_time"]
                    task["mem_obs_pose"] = task["current_obs_pose"]
            
                success, (start_t, end_t, retrieval_t) = evaluate_one_retrieval_task(args, agent, task)
                result = {
                    "task": task["task"],
                    "task_type": task_type,
                    "instance_name": task["instance_name"],
                    "instance_class": task["instance_class"],
                    "gt_start_time": start_t,
                    "gt_end_time": end_t,
                    "retrieval_time": retrieval_t,
                    "success": success,
                }
                results[task_type].append(result)
                
                # Update success rate display
                pbar.update(1)
                success = sum([r["success"] for r in results[task_type]])
                total = len(results[task_type])
                rate = 100.0 * success / total if total > 0 else 0.0
                pbar.set_postfix_str(f"({rate:.1f}%)")
            
            utility.drop_collection(db_name)
            import time; time.sleep(1)
    pbar.close()
    
    output["results"] = results
    return output
            
if __name__ == "__main__":
    args = parse_args()
    results = evaluate(args)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Evaluation results saved to {args.output_file}")
    
    # Print summary
    print("📊 Evaluation Summary:")
    for task_type, result_list in results["results"].items():
        total = len(result_list)
        success = sum(r["success"] for r in result_list)
        rate = 100.0 * success / total if total > 0 else 0.0
        print(f" - {task_type:<20}: {success}/{total} succeeded ({rate:.1f}%)")