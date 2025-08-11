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
from agent.utils.memloader import remember
from agent.utils.skills import *

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
        default="evaluation/sim_outputs/",
        help="Directory to save evaluation results (default: evaluation/sim_outputs/mem_retrieval/)"
    )
    parser.add_argument(
        "--task_file",
        type=str,
        default="evaluation/config/tasks_sim.txt",
        help="Path to the task file.",
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs="*",
        default=["unambiguous", "spatial", "spatial_temporal"],
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
        eval=True,
        class_type=task["instance_class"],
    )
    result = full_result["task_result"]
    if result is None:
        return {"result": {
                    "success": False,
                    "reference_resolution_successs": False,
                    "reference_resoluation_path": "",
                    "retrieval_grounding_success": False,
                    "retrieval_grounding_path": "",
                    "latest_retrieval_success": False,
                    "last_known_state_success": False,
                    "position": None,
                    "theta": None,
                    "has_picked": False,
                    "retrieved_instance_name": ""
                },
                "reasoning_toolcalls": full_result.get("search_in_time_toolcalls", [])}

    if result.has_picked is None or result.instance_name is None:
        return {"result": {
                    "success": False,
                    "reference_resolution_successs": False,
                    "reference_resoluation_path": "",
                    "retrieval_grounding_success": False,
                    "retrieval_grounding_path": "",
                    "latest_retrieval_success": False,
                    "last_known_state_success": False,
                    "position": None,
                    "theta": None,
                    "has_picked": False,
                    "retrieved_instance_name": ""
                },
                "reasoning_toolcalls": full_result.get("search_in_time_toolcalls", [])}
    success = task["instance_name"].lower() in result.instance_name.lower()
    
    ref_record = result.reference_resolution_records
    ret_record = result.retrieval_grounding_records
    
    # Extract datanames from records
    ref_dataname = extract_dataname_from_vidpath(ref_record["vidpath"]) if ref_record is not None else None
    ret_dataname = extract_dataname_from_vidpath(ret_record["vidpath"]) if ret_record is not None else None
    
    reference_resolution_successs, retrieval_grounding_successs, latest_retrieval_successs, last_known_state_success = False, False, False, False
    if ref_record is not None:
        for frame_idx in range(ref_record["start_frame"], ref_record["end_frame"]+1):
            if frame_idx >= len(annotations[ref_dataname]["frames"]):
                continue
            annotation = annotations[ref_dataname]["frames"][frame_idx]
            all_visible_instances = [x["prefab_name"] for x in annotation["frame_nodes"]]
            if task["instance_name"] in all_visible_instances:
                reference_resolution_successs = True
                break
    if ret_record is not None:
        for frame_idx in range(ret_record["start_frame"], ret_record["end_frame"]+1):
            if frame_idx >= len(annotations[ref_dataname]["frames"]):
                continue
            annotation = annotations[ret_dataname]["frames"][frame_idx]
            all_visible_instances = [x["prefab_name"] for x in annotation["frame_nodes"]]
            if task["instance_name"] in all_visible_instances:
                retrieval_grounding_successs = True
                break
        if retrieval_grounding_successs:
            retrieved_unix_time = ret_record["timestamp"]
            if abs(retrieved_unix_time - task["lastest_unix_time"]) < 3600:
                latest_retrieval_successs = True
            else:
                object_placements = annotations[ret_dataname]["object_placements"]
                gt_object_placements = annotations[exec_dataname]["object_placements"]
                
                selected_obj_placement = None
                for placement in object_placements:
                    if placement["obj_prefab_name"] == task["instance_name"]:
                        selected_obj_placement = placement
                        break
                gt_obj_placement = None
                for placement in gt_object_placements:
                    if placement["obj_prefab_name"] == task["instance_name"]:
                        gt_obj_placement = placement
                        break
                if selected_obj_placement is None or gt_obj_placement is None:
                    latest_retrieval_successs = False
                else:
                    latest_retrieval_successs = selected_obj_placement["surface_id"] == gt_obj_placement["surface_id"]
    
    image_path_fn = lambda vidpath, start_f, end_f: os.path.join(vidpath, f"{(start_f + end_f) // 2:06d}.png")
    reference_resoluation_path = image_path_fn(ref_record["vidpath"], 
                                               ref_record["start_frame"], 
                                               ref_record["end_frame"]) if ref_record is not None else None
    retrieval_grounding_path = image_path_fn(ret_record["vidpath"], 
                                             ret_record["start_frame"], 
                                             ret_record["end_frame"]) if ret_record is not None else None
    
    last_known_state_success = task["instance_name"] in result.visible_instances
    
    return {
        "result": {
            "success": success,
            "reference_resolution_successs": reference_resolution_successs,
            "reference_resoluation_path": reference_resoluation_path,
            "retrieval_grounding_success": retrieval_grounding_successs,
            "retrieval_grounding_path": retrieval_grounding_path,
            "latest_retrieval_success": latest_retrieval_successs,
            "last_known_state_success": last_known_state_success,
            "position": result.position,
            "theta": result.theta,
            "has_picked": result.has_picked,
            "retrieved_instance_name": result.instance_name
        },
        "reasoning_toolcalls": full_result.get("search_in_time_toolcalls", []),
    }
    
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
    
    if "gt" in args.agent_type:
        args.prompt_type = "gt"
        data_metadata = load_virtualhome_data_metadata(args.data_dir, caption_type="gt")
    elif "caption" in args.agent_type:
        args.prompt_type = "caption"
        data_metadata = load_virtualhome_data_metadata(args.data_dir, caption_type="nframe1")
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}. Supported types are 'low_level_gt' and 'high_level_gt'.")
    versions = [""]
    task_metadata = load_task_metadata(
        args.task_file, 
        args.benchmark_dir, 
        args.task_types,
        prefix="sim_tasks",
        versions=versions
    )
    included_task_types = task_metadata.keys()
    annotations = load_virtualhome_annotations(args.data_dir)
    
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
                
            if "high_level" in args.agent_type:
                from agent.agent_highlevel import HighLevelAgent
                agent = HighLevelAgent(
                    promt_type=args.prompt_type,
                    navigate_fn=navigate,
                    find_object_fn=find_object,
                    pick_fn=pick_by_instance_id,
                    logdir=result_dir,
                    logger_prefix=args.agent_type
                )
            elif "low_level" in args.agent_type:
                from agent.agent_lowlevel import LowLevelAgent
                agent = LowLevelAgent(
                    prompt_type=args.prompt_type,
                    navigate_fn=navigate,
                    find_object_fn=find_object,
                    pick_fn=pick_by_instance_id,
                    logdir=result_dir,
                    logger_prefix=args.agent_type
                )
            else:
                raise ValueError(f"Unknown agent type: {args.agent_type}. Supported types are 'high_level' and 'low_level'.")
            
            with open(task_path, "r") as f:
                task_data = json.load(f)
                
            db_name = f"virtualhome_{task_type}_{task_id}"
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            obs_savepath = f"data/cobot/{task_type}_{task_id}_{timestamp}/"
            memory = MilvusMemory(db_name, obs_savepth=obs_savepath)
            memory.reset()
    
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
    
            inpaths, time_offsets, viddirs = [], [], []
            for bi, bagname in enumerate(task_data["bagnames"]):
                inpaths.append(data_metadata[bagname])
                time_offsets.append(bag_unix_times[bi])
                viddirs.append(os.path.join(args.data_dir, bagname, "images"))
            remember(memory, inpaths, time_offsets, viddirs)
            
            agent.set_memory(memory)
            
            exec_bagname = task_data["bagnames"][-1]
            
            for task in task_data["tasks"]:
                # Reset the testing environment
                graph_path = os.path.join(args.data_dir, exec_bagname, "0", "graph.json")
                scene_id = int(exec_bagname.split("_")[0].replace("scene", ""))
                if not set_virtulhome_scene(graph_path, scene_id):
                    import pdb; pdb.set_trace()
                
                _, (lastest_date_str, lastest_time_str) = task_data["bag_time_mapping"][-1]
                dt = datetime.strptime(f"{lastest_date_str} {lastest_time_str}", "%Y-%m-%d %H:%M")
                dt = dt.replace(tzinfo=timezone.utc)
                lastest_unix_time = dt.timestamp()
                
                task["lastest_unix_time"] = lastest_unix_time
                
                full_result = evaluate_one_task(agent, task, annotations, exec_bagname)
                result = full_result["result"]
                result["task"] = task["task"]
                result["task_type"] = task_type
                result["instance_name"] = task["instance_name"]
                result["instance_class"] = task["instance_class"]
                
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)
                    
                reasoning_toolcalls_path = os.path.join(result_dir, f"reasoning_toolcalls_{args.agent_type}_{task_id}.json")
                reasoning_toolcalls = full_result.get("reasoning_toolcalls", [])
                with open(reasoning_toolcalls_path, "w") as f:
                    json.dump(reasoning_toolcalls, f, indent=2)
                
                reasoning_toolcalls_path = os.path.join(result_dir, f"reasoning_toolcalls_{args.agent_type}_{task_id}.txt")
                ARG_ORDER = ["x", "time", "position", "record_id", "record_ids", "start_time", "end_time", "k"]
                with open(reasoning_toolcalls_path, "w") as f:
                    for toolcall in reasoning_toolcalls:
                        toolname = toolcall.get("name", "unknown_tool")
                        all_toolargs = toolcall.get("args", {})
                        toolargs = [str(all_toolargs[arg]) for arg in ARG_ORDER if arg in all_toolargs]
                        f.write(f'{toolname}({", ".join(toolargs)})\n')
                
                before_one_task_finish(results, result, pbar)
                
            # Clean up memory and agent state
            agent.flush_tool_threads()
            import time; time.sleep(1)
            if utility.has_collection(db_name):
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
        
        with open(output_path, "w") as f:
            json.dump(result_list, f, indent=2)
        print(f"✅ Saved {task_type} results to {output_path}")

    # Print summary
    print("📊 Evaluation Summary:")
    for task_type, result_list in results["results"].items():
        total = len(result_list)
        success = sum(r["success"] for r in result_list)
        rate = 100.0 * success / total if total > 0 else 0.0
        summary = f" - {task_type:<20}: {success}/{total} succeeded ({rate:.1f}%)"
        print(summary)