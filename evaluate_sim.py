import argparse
from datetime import datetime, timezone
from pathlib import Path
from pymilvus import utility
from collections import defaultdict
from tqdm import tqdm

from evaluation.eval_utils import *
# from agent.agent import Agent
# from agent.agent2 import Agent
from agent.agent_lowlevel import LowLevelAgent
from memory.memory import MilvusMemory, MemoryItem
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
        "--include_common_sense",
        action='store_true',
        help="Whether to include common sense reasoning in the evaluation.",
    )
    parser.add_argument(
        "--include_recaption",
        action='store_true',
        help="Whether to include recaptioning in the evaluation.",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        help="Type of agent to use (e.g., 'low_level', 'high_level').",
    )
    args = parser.parse_args()
    return args

def evaluate_one_task(agent, task: dict):
    result = agent.run(
        question=task['task'],
    )

    if result.has_picked is None or result.instance_name is None:
        return (False, False, None)
    obj_retrieval_success = task["instance_name"].lower() in result.instance_name.lower()
    mem_retrieval_success = obj_retrieval_success
    return (obj_retrieval_success, mem_retrieval_success, result.instance_name)
    
def evaluate(args):
    if args.agent_type == "high_level":
        from agent.agent_highlevel import HighLevelAgent
        agent = HighLevelAgent(
            navigate_fn=navigate,
            find_object_fn=find_object,
            pick_fn=pick_by_instance_id,
        )
    elif args.agent_type == "low_level":
        from agent.agent_lowlevel import LowLevelAgent
        agent = LowLevelAgent(
            navigate_fn=navigate,
            find_object_fn=find_object,
            pick_fn=pick_by_instance_id,
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}. Supported types are 'high_level' and 'low_level'.")
    
    data_metadata = load_virtualhome_data_metadata(args.data_dir)
    versions = [""]
    task_metadata = load_task_metadata(
        args.task_file, 
        args.benchmark_dir, 
        args.task_types,
        prefix="sim_tasks",
        versions=versions
    )
    included_task_types = task_metadata.keys()
    
    output = {
        "task_metadata": task_metadata,
    }
    results = defaultdict(list)
    
    for task_type, task_paths in task_metadata.items():
        if task_type not in included_task_types:
            continue
        
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
            
            for task in task_data["tasks"]:
                # Reset the testing environment
                graph_path = os.path.join(args.data_dir, bagname, "0", "graph.json")
                scene_id = int(bagname.split("_")[0].replace("scene", ""))
                if not set_virtulhome_scene(graph_path, scene_id):
                    import pdb; pdb.set_trace()
                
                obj_success, mem_success, retrieved_instance = evaluate_one_task(agent, task)
                
                result = {
                    "task": task["task"],
                    "task_type": task_type,
                    "instance_name": task["instance_name"],
                    "instance_class": task["instance_class"],
                    "success": obj_success,
                    "mem_success": mem_success or obj_success,  # If object retrieval is successful, memory retrieval is also considered successful
                    "retrieved_instance": retrieved_instance,
                    "target_instance": task["instance_name"],
                } # TODO figure out why ground truth sometimes missed the target instance
                
                results[task_type].append(result)
                pbar.update(1)
                success = sum([r["success"] for r in results[task_type]])
                total = len(results[task_type])
                rate = 100.0 * success / total if total > 0 else 0.0
                pbar_postfix_str = f"({rate:.1f}%)"
                pbar.set_postfix_str(pbar_postfix_str)
                
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
    
    results = evaluate(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
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