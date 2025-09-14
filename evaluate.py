import argparse
from datetime import datetime, timezone
from pathlib import Path
from pymilvus import utility
from collections import defaultdict
from tqdm import tqdm
import glob
import json
import os

from evaluation.eval_utils import *
from memory.memory import MilvusMemory
from agent.utils.memloader import remember_tiago

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the agent in simulation.")
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
        "--output_dir",
        type=str,
        default="evaluation/outputs/",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--task_file",
        type=str,
        default="evaluation/config/tasks.txt",
        help="Path to the task file.",
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs="*",
        default=["classonly", "unambiguous", "spatial", "spatial_temporal", "frequency", "common_sense"],
        help="List of task type prefixes to evaluate (e.g., unambiguous spatial). If not set, evaluate all."
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        help="Type of agent to use (e.g., 'low_level', 'high_level').",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Whether to force rerun the evaluation. If True, will rerun all tasks even if they have already been evaluated.",
    )
    args = parser.parse_args()
    return args

def extract_dataname_from_vidpath(vidpath: str) -> str:
    """
    Extract dataname from vidpath.
    Example: '...unity_output/scene4_02/images' -> 'scene4_02'
    """
    if vidpath is None:
        return None
    
    # Split the path and find the directory before 'images'
    path_parts = vidpath.split('/')
    try:
        # Find the index of 'images' and get the directory before it
        images_index = path_parts.index('images')
        if images_index > 0:
            return path_parts[images_index - 1]
        else:
            return None
    except ValueError:
        # 'images' not found in path
        return None

def evaluate_one_task(agent, task: dict, annotations: dict, exec_dataname: str):
    full_result = agent.run(
        question=task['task'],
    )
    ret_result = {
        "success": False,
        "position": None,
        "theta": None,
        "searched_poses": [],
    }
    
    result = full_result["task_result"]
    
    if result.position[0] == -1 and result.position[1] == -1 and result.position[2] == -1 and result.theta == -1:
        success = False
    else:
        wp = match_ahg_waypoint(result.position, result.theta)
        success = (task["waypoint"] == wp)
        
        ret_result["success"] = success
        ret_result["position"] = result.position
        ret_result["theta"] = result.theta

    ret_result["searched_poses"] = result.searched_poses if hasattr(result, "searched_poses") else []

    return {"result": ret_result, "toolcalls": full_result.get("toolcalls", [])}

def evaluate(args):
    
    def before_one_task_finish(results, result, pbar):
        results[task_type].append(result)
        pbar.update(1)
        success = sum([r["success"] for r in results[task_type]])
        total = len(results[task_type])
        rate = 100.0 * success / total if total > 0 else 0.0
        pbar_postfix_str = f"({rate:.1f}%)"
        pbar.set_postfix_str(pbar_postfix_str)
        return results, pbar
    
    if "caption" in args.agent_type:
        args.prompt_type = "caption"
        data_metadata = load_ahg_data_metadata(args.data_dir)
    elif args.agent_type == "random" or args.agent_type == "sg":
        # TODO caption type is a dummy
        data_metadata = load_ahg_data_metadata(args.data_dir)
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}. Supported types are 'low_level_gt' and 'high_level_gt'.")
    versions = [""]
    task_metadata = load_task_metadata(
        args.task_file, 
        args.benchmark_dir, 
        args.task_types,
        prefix="tasks",
        versions=versions
    )
    included_task_types = task_metadata.keys()
    
    annotation_file = os.path.join(args.benchmark_dir, "annotations", "movable_objects.json")
    annotations = load_ahg_annotations(annotation_file)

    output = {
        "task_metadata": task_metadata,
    }
    results = defaultdict(list)
    
    if args.force_rerun:
        for task_type, task_paths in task_metadata.items():
            for task_path in task_paths:
                task_id = Path(task_path).stem
                results_dir = os.path.join(args.output_dir, task_type, task_id)
                # Patterns to match
                patterns = [
                    f"results_{args.agent_type}_*.json",
                    f"{args.agent_type}_*.log"
                ]
                if os.path.exists(results_dir):
                    for pattern in patterns:
                        files_to_delete = glob.glob(os.path.join(results_dir, pattern))
                        for f in files_to_delete:
                            try:
                                os.remove(f)
                                print(f"🗑️ Deleted: {f}")
                            except Exception as e:
                                print(f"⚠️ Failed to delete {f}: {e}")
    
    for task_type, task_paths in task_metadata.items():
        if task_type not in included_task_types:
            continue
        
        os.makedirs(os.path.join(args.output_dir, task_type), exist_ok=True)
        
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
            
            result_dir = os.path.join(args.output_dir, task_type, task_id)
            os.makedirs(result_dir, exist_ok=True)
            
            result_path = os.path.join(result_dir, f"results_{args.agent_type}_{task_id}.json")
            if not args.force_rerun and os.path.exists(result_dir) and os.path.exists(result_path):
                with open(result_path, "r") as f:
                    result = json.load(f)
                if result is not None and "success" in result:
                    before_one_task_finish(results, result, pbar)
                    continue
                
            if "low_level" in args.agent_type and "replan" in args.agent_type:
                from agent.agent_lowlevel_replan import ReplanLowLevelAgent
                agent = ReplanLowLevelAgent(
                    prompt_type=args.prompt_type,
                    logdir=result_dir,
                    logger_prefix=args.agent_type,
                    is_interactive=("interactive" in task_type),
                    robot_model="tiago"
                )
            elif "high_level" in args.agent_type:
                from agent.agent_highlevel import HighLevelAgent
                agent = HighLevelAgent(
                    prompt_type=args.prompt_type,
                    logdir=result_dir,
                    logger_prefix=args.agent_type,
                    is_interactive=("interactive" in task_type),
                    robot_model="tiago"
                )
            elif "low_level" in args.agent_type:
                from agent.agent_lowlevel import LowLevelAgent
                agent = LowLevelAgent(
                    prompt_type=args.prompt_type,
                    logdir=result_dir,
                    logger_prefix=args.agent_type,
                    is_interactive=("interactive" in task_type),
                    robot_model="tiago"
                )
            elif "random" == args.agent_type:
                from agent.agent_random import RandomAgent
                from agent.utils.skills import navigate, detect_virtual_home_object, pick_by_instance_id
                agent = RandomAgent(
                    navigate_fn=navigate,
                    detect_fn=detect_virtual_home_object,
                    pick_fn=pick_by_instance_id,
                    logdir=result_dir,
                    logger_prefix=args.agent_type
                )
            elif "sg" == args.agent_type:
                from agent.agent_scenegraph import SceneGraphAgent
                agent = SceneGraphAgent(
                    logdir=result_dir,
                    logger_prefix=args.agent_type,
                    is_interactive=("interactive" in task_type),
                    robot_model="tiago"
                )
            else:
                raise ValueError(f"Unknown agent type: {args.agent_type}. Supported types are 'high_level' and 'low_level'.")
            
            with open(task_path, "r") as f:
                task_data = json.load(f)
                
            bag_unix_times = []
            for bagname, (date_str, time_str) in task_data["bag_time_mapping"]:
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                    dt = dt.replace(tzinfo=timezone.utc)
                    unix_time = dt.timestamp()
                    bag_unix_times.append(unix_time)
                except Exception as e:
                    raise ValueError(f"Failed to parse datetime for bag '{bagname}': {e}")
    
            assert len(bag_unix_times) == len(task_data["bagnames"]), \
                f"Number of bag names ({len(task_data['bagnames'])}) does not match number of unix times ({len(bag_unix_times)})"
                
            mem_bagnames = task_data["bagnames"]
            if task_type == "common_sense":
                mem_bagnames = mem_bagnames[:-1]
                
            if args.agent_type == "sg":
                memory_sg = load_virtualhom_memory_sg(args.data_dir, mem_bagnames, bag_unix_times, is_common_sense=("common_sense" in task_type))
                agent.set_memory_sg(memory_sg)
                # print(len(__import__("tiktoken").encoding_for_model("gpt-4o").encode(memory_sg)))
                # import pdb; pdb.set_trace()
                
            elif args.agent_type != "random": # Set up memory
                db_name = f"virtualhome_{task_type}_{task_id}"
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                obs_savepath = f"data/cobot/{task_type}_{task_id}_{timestamp}/"
                memory = MilvusMemory(db_name, obs_savepth=obs_savepath)
                memory.reset()
    
                inpaths, time_offsets, viddirs = [], [], []
                for bi, bagname in enumerate(mem_bagnames):
                    inpaths.append(data_metadata[bagname])
                    time_offsets.append(bag_unix_times[bi])
                    viddirs.append(os.path.join(args.data_dir, bagname, "images"))
                remember_tiago(memory, inpaths, time_offsets, viddirs)
                agent.set_memory(memory)
                
            exec_bagname = task_data["bagnames"][-1]
            
            for task in task_data["tasks"]:
                # Reset the testing environment
                if not set_virtulhome_scene(os.path.join(args.benchmark_dir, "annotations", exec_bagname)):
                    import pdb; pdb.set_trace()
                    raise ValueError(f"Failed to set VirtualHome scene to {exec_bagname}")
                
                _, (lastest_date_str, lastest_time_str) = task_data["bag_time_mapping"][-1]
                dt = datetime.strptime(f"{lastest_date_str} {lastest_time_str}", "%Y-%m-%d %H:%M")
                dt = dt.replace(tzinfo=timezone.utc)
                lastest_unix_time = dt.timestamp()
                
                task["lastest_unix_time"] = lastest_unix_time
                
                if args.agent_type == "random":
                    poses = load_virtualhome_poses(args.data_dir, list(set(task_data["bagnames"])))
                    agent.before_run(task["instance_class"], poses)
                    
                full_result = evaluate_one_task(agent, task, annotations, exec_bagname)
                result = full_result["result"]
                result["task"] = task["task"]
                result["task_type"] = task_type
                result["instance_name"] = task.get("instance_name", "")
                result["instance_class"] = task["instance_class"]
                
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)
                    
                toolcalls_path = os.path.join(result_dir, f"toolcalls_{args.agent_type}_{task_id}.json")
                toolcalls = full_result.get("toolcalls", [])
                with open(toolcalls_path, "w") as f:
                    json.dump(toolcalls, f, indent=2)

                toolcalls_path = os.path.join(result_dir, f"toolcalls_{args.agent_type}_{task_id}.txt")
                ARG_ORDER = ["x", "time", "position", "record_id", "record_ids", "start_time", "end_time", "k", "pos", "theta", "query_text", "instance_id"]
                with open(toolcalls_path, "w") as f:
                    for toolcall in toolcalls:
                        toolname = toolcall.get("name", "unknown_tool")
                        all_toolargs = toolcall.get("args", {})
                        toolargs = [str(all_toolargs[arg]) for arg in ARG_ORDER if arg in all_toolargs]
                        f.write(f'{toolname}({", ".join(toolargs)})\n')
                
                before_one_task_finish(results, result, pbar)
                
            # Clean up memory and agent state
            agent.flush_tool_threads()
            import time; time.sleep(1)
            if args.agent_type != "random" and ("sg" not in args.agent_type) and utility.has_collection(db_name):
                utility.drop_collection(db_name)
            import time; time.sleep(1)
            
        pbar.close()
        
    output["results"] = results
    return output
    
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = evaluate(args)
    
    for task_type, result_list in results["results"].items():
        output_path = os.path.join(args.output_dir, f"results_{args.agent_type}_{task_type}.json")
        
        result = {
            "task_metadata": results["task_metadata"],
            "results": {},
        }
        result["results"][task_type] = result_list
        
    # Print summary
    print("📊 Evaluation Summary:")
    for task_type, result_list in results["results"].items():
        total = len(result_list)
        success = sum(r["success"] for r in result_list)
        rate = 100.0 * success / total if total > 0 else 0.0
        summary = f" - {task_type:<20}: {success}/{total} succeeded ({rate:.1f}%)"
        print(summary)