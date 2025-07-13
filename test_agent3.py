import datetime
import time
import argparse

from agent.agent2 import Agent
from memory.memory import MilvusMemory
from agent.utils.skills import *
from agent.utils.tools2 import (
    create_recall_best_match_tool, 
    create_memory_inspection_tool,
    create_determine_search_instance_tool,
    create_determine_unique_instances_tool
)
from agent.utils.function_wrapper import FunctionsWrapper
from agent.utils.debug import get_logger

def load_toy_memory(memory: MilvusMemory):
    ONE_DAY = 24 * 60 * 60  # seconds in o
    from datetime import datetime
    start_t = datetime.strptime("2025-07-08 19:30:00", "%Y-%m-%d %H:%M:%S").timestamp()
    
    records = [
        {
            "time": start_t+0.0,
            "base_position": [0.0, 0.0, 0.0],
            "base_caption": "I saw a cup",
            "start_frame": 0,
            "end_frame": 10
        },
        {
            "time": start_t+1.0,
            "base_position": [0.56, 0.0, 1.0],
            "base_caption": "I saw a table",
            "start_frame": 11,
            "end_frame": 20
        },
        {
            "time": start_t+2.0,
            "base_position": [1.1, 1.0, 1.0],
            "base_caption": "I saw a chair",
            "start_frame": 21,
            "end_frame": 30
        },
        {
            "time": start_t+3.0,
            "base_position": [1.1, 1.0, 1.0],
            "base_caption": "I saw a cup",
            "start_frame": 31,
            "end_frame": 40
        },
        {
            "time": ONE_DAY+start_t+0.0,
            "base_position": [0.0, 0.1, 0.0],
            "base_caption": "I saw a cup",
            "start_frame": 41,
            "end_frame": 50
        },
        
    ]
 
    for record in records:
        embedding = memory.embedder.embed_query(record["base_caption"])
        memory_item = MemoryItem(
            caption=record["base_caption"],
            text_embedding=embedding,
            time=record["time"],
            position=record["base_position"],
            theta=0.0,  # Assuming theta is not used in this toy example
            vidpath="debug/toy_examples",
            start_frame=record["start_frame"],
            end_frame=record["end_frame"]
        )
        memory.insert(memory_item)
        
if __name__ == "__main__":
    memory = MilvusMemory("test", obs_savepth=None)
    load_toy_memory(memory)
    
    outputs = memory.search_by_position_and_time([0,0.1,0], "2025-07-08 19:30:00", "2025-07-08 20:30:00", k=2)
    print(outputs)
    
    outputs = memory.search_by_time("2025-07-08 19:30:00", k=1)
    print(outputs)