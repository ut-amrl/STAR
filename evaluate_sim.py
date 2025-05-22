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
        "--data_dir",
        type=str,
        # required=True,
        help="Path to the directory containing the data files.",
    )
    args = parser.parse_args()
    return args

def evaluate_one_task(args, agent: Agent, task: dict):
    result = agent.run(
        question=task['task'],
        today=f"Today is {args.current_pretty_date}."
    )
    
def evaluate(args):
    agent = Agent(
        navigate_fn=navigate,
        observe_fn=observe,
        pick_fn=pick,
        image_path_fn=get_image_path_for_simulation,
    )
    
    questions = [
        "Bring me the green book.",
    ]
    
    db_name = "test_virtualhome"
    memory = MilvusMemory(db_name, obs_savepth=None)
    memory.reset()
    
    inpaths = [
        "/robodata/taijing/benchmarks/virtualhome/unity_output/scene4_754ab231d3_0/0/caption_gpt4o.json",
    ]
    # Convert "2025-02-27 09:00:00" to double UNIX timestamp
    dt = datetime(2025, 2, 27, 9, 0, 0, tzinfo=timezone.utc)
    timestamp = float(dt.timestamp())
    time_offsets = [
        timestamp,
    ]
    try:
        args.current_pretty_date = dt.strftime("%b %-d, %Y")  # Linux/Mac
    except ValueError:
        args.current_pretty_date = dt.strftime("%b %#d, %Y")  # Windows fallback
    viddirs = [
        "/robodata/taijing/benchmarks/virtualhome/unity_output/scene4_754ab231d3_0/0/",
    ]
    remember(memory, inpaths, time_offsets, viddirs)
    agent.set_memory(memory)
    
    for question in questions:
        task = {
            "task": question,
        }
        evaluate_one_task(args, agent, task)
    
    utility.drop_collection(db_name)
    import time; time.sleep(1)

if __name__ == "__main__":
    args = parse_args()
    results = evaluate(args)