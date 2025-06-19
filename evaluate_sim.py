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
        "--eval_type",
        type=str,
        required=True,
        help="Evaluation type to run (e.g., 'mem_retrieval', 'execution')."
    )
    parser.add_argument(
        "--include_common_sense",
        action='store_true',
        help="Whether to include common sense reasoning in the evaluation.",
    )
    args = parser.parse_args()
    return args

def evaluate_one_mem_retrieval_task(args, agent: Agent, task: dict, annotations):
    result = agent.run(
        question=task['task'],
        today=f"Today is {args.current_pretty_date}.",
        graph_type="retrieval",
    )
    if result is None:
        return (False, (None, None, None, None))
    
    instances = []
    for annotation in annotations:
        if int(annotation["start_frame"]) >= int(result["start_frame"]) and \
           int(result["end_frame"]) >= int(annotation["end_frame"]):
               instances += [item for sublist in annotation["target_instances"].values() for item in sublist]
    instances = list(set(instances))
    
    return ((task["instance_name"] in instances) and (task["target_bagname_memory"] in result["vidpath"]), 
            (result["start_frame"], result["end_frame"], task["instance_name"], instances))
    
def evaluate_one_execution_task(args, agent: Agent, task: dict, annotations):
    try:
        result = agent.run(
            question=task['task'],
            today=f"Today is {args.current_pretty_date}.",
            graph_type="no_replanning"
        )
    except Exception as e:
        import pdb; pdb.set_trace()
        return (False, False, None)
    
    retrieved_record = result.curr_target()
    if retrieved_record is None:
        mem_retrieval_success = False
    elif retrieved_record["id"] == -1:
        mem_retrieval_success = False
    else:
        instances = []
        for annotation in annotations:
            if int(annotation["start_frame"]) >= int(retrieved_record["start_frame"]) and \
            int(retrieved_record["end_frame"]) >= int(annotation["end_frame"]):
                instances += [item for sublist in annotation["target_instances"].values() for item in sublist]
        instances = set(instances)
        if task["task_type"] == "classonly":
            mem_retrieval_success = False
            for instance in instances:
                if task["instance_name"] in instance.lower():
                    mem_retrieval_success = True
                    break
            mem_retrieval_success = mem_retrieval_success and (task["target_bagname_memory"] in retrieved_record["vidpath"])
        else:
            mem_retrieval_success = (task["instance_name"] in instances) and (task["target_bagname_memory"] in retrieved_record["vidpath"])
    
    if not result.has_picked:
        obj_retrieval_success = False
        retrieved_instance = None
    else:
        if task["task_type"] == "classonly":
            obj_retrieval_success = task["instance_name"] in result.instance_uid.lower()
        else:
            obj_retrieval_success = (task["instance_name"] == result.instance_uid)
        retrieved_instance = result.instance_uid
        
    if mem_retrieval_success and (not obj_retrieval_success) and len(instances) == 1:
        import pdb; pdb.set_trace()
        
    return (obj_retrieval_success, mem_retrieval_success, retrieved_instance)
    
def evaluate(args):
    agent = Agent(
        agent_type="sim",
        navigate_fn=navigate,
        find_object_fn=find_object,
        observe_fn=observe,
        pick_fn=pick,
        image_path_fn=get_image_path_for_simulation,
    )
    
    data_metadata = load_virtualhome_data_metadata(args.data_dir)
    versions = [""]
    if args.include_common_sense:
        versions += ["_common_sense"]
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
    
            bag_unix_times = {}
            for bagname, (date_str, time_str) in task_data["bag_time_mapping"].items():
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                    if bagname == task_data["bagnames"][-1]:
                        try:
                            args.current_pretty_date = dt.strftime("%A, %b %-d, %Y")  # Linux/Mac
                        except ValueError:
                            args.current_pretty_date = dt.strftime("%A, %b %#d, %Y")  # Windows fallback
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
            remember(memory, inpaths, time_offsets, viddirs)
            
            with open(os.path.join(str(Path(inpaths[-1]).parent), "caption_synthetic.json"), "r") as f:
                annotations = json.load(f)
            if annotations is None:
                raise ValueError(f"No annotations found in {inpaths[-1]}")
                
            agent.allow_recaption = "recaption" in task_type
            agent.allow_common_sense = "common_sense" in task_type
            agent.set_memory(memory)
            
            for task in task_data["tasks"]:
                # Reset the testing environment
                graph_path = os.path.join(args.data_dir, bagname, "0", "graph.json")
                scene_id = int(bagname.split("_")[0].replace("scene", ""))
                if not set_virtulhome_scene(graph_path, scene_id):
                    import pdb; pdb.set_trace()
                
                # TODO use args.eval_types to determine which tasks to run
                if args.eval_type == "mem_retrieval":
                    success, _ = evaluate_one_mem_retrieval_task(args, agent, task, annotations)
                    result = {
                        "task": task["task"],
                        "task_type": task_type,
                        "instance_name": task["instance_name"],
                        "instance_class": task["instance_class"],
                        "success": success,
                    }
                elif args.eval_type == "execution":
                    obj_success, mem_success, retrieved_instance = evaluate_one_execution_task(args, agent, task, annotations)
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
                else:
                    raise ValueError(f"Unknown evaluation type: {args.eval_type}")
                
                results[task_type].append(result)
                pbar.update(1)
                success = sum([r["success"] for r in results[task_type]])
                total = len(results[task_type])
                rate = 100.0 * success / total if total > 0 else 0.0
                pbar_postfix_str = f"({rate:.1f}%)"
                if args.eval_type == "execution":
                    mem_success = sum([r["mem_success"] for r in results[task_type]])
                    mem_rate = 100.0 * mem_success / total if total > 0 else 0.0
                    pbar_postfix_str += f"/({mem_rate:.1f})%"
                pbar.set_postfix_str(pbar_postfix_str)
                
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
        output_path = os.path.join(args.output_dir, f"results_{task_type}.json")
        
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
        if args.eval_type == "execution":
            mem_success = sum(r["mem_success"] for r in result_list)
            mem_rate = 100.0 * mem_success / total if total > 0 else 0.0
            summary += f", Memory Retrieval: {mem_success}/{total} ({mem_rate:.1f}%)"
        print(summary)