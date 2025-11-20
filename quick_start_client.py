import os
import json
from pathlib import Path

from agent.agent_lowlevel_replan import STARAgent
from memory.memory import MilvusMemory, MemoryItem

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Quick Start for STAR")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/qs_outputs/",
        help="Directory to save evaluation results (default: evaluation/qs_outputs/)"
    )
    parser.add_argument(
        "--captioner",
        type=str,
        default="openai",
        help="Captioner: 'openai' or 'molmo'"
    )
    args = parser.parse_args()
    return args

def remember_demo(memory: MilvusMemory, captioner_t: str, dataname: str = "toy_data_1"):
    tag = "molmo" if captioner_t == "molmo" else f"gpt4o"
    inpath = os.path.join("example", dataname, f"caption_{tag}_nframe1.json")
    
    with open(inpath, 'r') as f:
        for entry in json.load(f):
            t, pos, theta, caption, text_embedding, start_frame, end_frame = entry["time"], entry["base_position"], entry["base_orientation"], entry["base_caption"], entry["base_caption_embedding"], entry["start_frame"], entry["end_frame"]
            
            # handle pos
            if len(pos) == 2:
                pos += [0.0]
            pos = [round(p, 1) for p in pos]
            
            # handle video path
            viddir = os.path.join("example", dataname)
            # vidpath = os.path.join(viddir, Path(inpath).stem)
            
            # insertion
            memory_item = MemoryItem(
                caption=caption,
                text_embedding=text_embedding,
                time=t,
                position=pos,
                theta=theta,
                vidpath=viddir,
                start_frame=start_frame,
                end_frame=end_frame
            )
            memory.insert(memory_item)
            
if __name__ == "__main__":
    args = parse_args()

    # Initialize STAR Agent
    agent = STARAgent(logdir=args.output_dir, logger_prefix="STAR")
    # Initialize Milvus Memory # TODO remove obs_savepath
    memory = MilvusMemory("qs_memory", obs_savepth="data/cobot/test")
    remember_demo(memory, args.captioner)
    
    agent.set_memory(memory)
    # result = agent.run("Bring me a book")["task_result"]
    result = agent.run("Bring me the book that was on the bookshelf this Monday morning")["task_result"]
    print("Navigated to ", (result.position, result.theta))
    print("Object picked: ", result.has_picked, result.instance_name)
    print("Searched poses:", result.searched_poses if hasattr(result, "searched_poses") else [])
    
    
    