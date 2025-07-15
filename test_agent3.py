import datetime
import time
import argparse
from datetime import datetime, timezone

from agent.agent_lowlevel import *
from memory.memory import MilvusMemory, MemoryItem
from agent.utils.skills import *
from agent.utils.tools2 import (
    create_recall_best_match_tool, 
    create_memory_inspection_tool,
    create_determine_search_instance_tool,
    create_determine_unique_instances_tool
)
from agent.utils.function_wrapper import FunctionsWrapper
from agent.utils.debug import get_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-memory', action='store_true', help='Run memory tests')
    parser.add_argument('--test-agent', action='store_true', help='Run agent tests')
    return parser.parse_args()

def load_toy_memory(memory: MilvusMemory):
    ONE_DAY = 24 * 60 * 60
    start_t = datetime.strptime("2025-07-08 19:30:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    
    records = [
        {"time": start_t+0.0, "base_position": [0.0, 0.0, 0.0], "base_caption": "I saw a cup", "start_frame": 0, "end_frame": 10},
        {"time": start_t+1.0, "base_position": [0.56, 0.0, 1.0], "base_caption": "I saw a table", "start_frame": 11, "end_frame": 20},
        {"time": start_t+2.0, "base_position": [1.1, 1.0, 1.0], "base_caption": "I saw a chair", "start_frame": 21, "end_frame": 30},
        {"time": start_t+3.0, "base_position": [1.1, 1.0, 1.0], "base_caption": "I saw a cup", "start_frame": 31, "end_frame": 40},
        {"time": ONE_DAY+start_t, "base_position": [0.0, 0.1, 0.0], "base_caption": "I saw a cup", "start_frame": 41, "end_frame": 50},
    ]
 
    for record in records:
        embedding = memory.embedder.embed_query(record["base_caption"])
        memory_item = MemoryItem(
            caption=record["base_caption"],
            text_embedding=embedding,
            time=record["time"],
            position=record["base_position"],
            theta=0.0,
            vidpath="debug/toy_examples",
            start_frame=record["start_frame"],
            end_frame=record["end_frame"]
        )
        memory.insert(memory_item)

def test_txt_search(memory):
    print("\n--- test_txt_search ---")
    print(memory.search_by_txt_and_time("cup", "2025-07-08 19:30:00", "2025-07-09 19:30:00", k=3))
    print(memory.search_by_txt_and_time("cat", "2025-07-08 19:30:00", "2025-07-09 19:30:00", k=1))
    print(memory.search_by_txt_and_time("cup", "2025-07-08 19:30:00", "2025-07-09 19:30:00"))
    print(memory.search_by_txt_and_time("cup", start_time="2025-07-08 19:30:00", k=1))
    print(memory.search_by_txt_and_time("cup", end_time="2025-07-08 19:30:00"))

def test_pos_search(memory):
    print("\n--- test_pos_search ---")
    print(memory.search_by_position_and_time([0, 0, 0], "2025-07-08 19:30:00", "2025-07-08 20:30:00", k=2))
    print(memory.search_by_position_and_time([10, 10, 10], "2025-07-08 19:30:00", "2025-07-08 20:30:00", k=2))

def test_time_search(memory):
    print("\n--- test_time_search ---")
    print(memory.search_by_time("2025-07-08 19:30:00", k=1))
    print(memory.search_by_time("2030-01-01 00:00:00", k=1))
    
def test_count_records_by_time(memory):
    print("\n--- test_count_records_by_time ---")
    # Full range: should include all inserted records (5 total)
    count = memory.count_records_by_time("2025-07-08 19:30:00", "2025-07-10 19:30:00")
    print(f"Count (full range): {count}")  # Expect: 5

    # Narrow range: only the first 3 (cup, table, chair)
    count = memory.count_records_by_time("2025-07-08 19:30:00", "2025-07-08 19:30:02")
    print(f"Count (first 3 records): {count}")  # Expect: 3

    # Range with only the last cup
    count = memory.count_records_by_time("2025-07-09 19:30:00", "2025-07-09 19:31:00")
    print(f"Count (last cup only): {count}")  # Expect: 1

    # Empty range
    count = memory.count_records_by_time("2025-07-11 00:00:00", "2025-07-11 01:00:00")
    print(f"Count (no records): {count}")  # Expect: 0


if __name__ == "__main__":
    args = parse_args()
    memory = MilvusMemory("test", obs_savepth=None)
    load_toy_memory(memory)

    if args.test_memory:
        test_count_records_by_time(memory)
        test_txt_search(memory)
        test_pos_search(memory)
        test_time_search(memory)
        
    if args.test_agent:
        agent = Agent()
        agent.set_memory(memory)
        task_metadata = {
            # "task_desc": "Find me a cup",
            # "task_desc": "Find me a white cup",
            # "task_desc": "Bring me a cup you saw yesterday.",
            "task_desc": "Bring me a white cup you saw yesterday.",
            # "task_desc": "Bring me my favorite cup.",
            "today_str": "2025-07-09"
        }
        agent.run(question=task_metadata["task_desc"], 
                today=task_metadata["today_str"], 
                graph_type="")
    