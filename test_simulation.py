import datetime
import time
import argparse

from agent.agent import Agent
from memory.memory import MilvusMemory
from agent.utils.memloader import remember
from agent.utils.skills import *

def parse_args():
    parser = argparse.ArgumentParser(description="Test the agent.")
    parser.add_argument(
        "--db_name",
        type=str,
        default="test_db",
        help="Name of the Milvus database to use (default: test_db)"
    )
    parser.add_argument(
        "--inpaths",
        type=str,
        nargs='+',
        required=True,
        help="List of caption files (required)"
    )
    parser.add_argument(
        "--viddirs",
        type=str,
        nargs='+',
        required=True,
        help="List of video directories (required)"
    )
    parser.add_argument(
        "--allow_recaption",
        action="store_true",
        help="Allow recaptioning (default: False)"
    )
    return parser.parse_args()

def run(args):
    agent = Agent(
        navigate_fn=navigate,
        observe_fn=observe,
        pick_fn=pick,
        visible_objects_fn=get_visible_objects,
    )
    memory = MilvusMemory(args.db_name, obs_savepth="data/cobot/test")
    
    today = datetime.date(2025, 2, 3)
    time_offsets = []
    for i in range(len(args.inpaths)):
        day = today - datetime.timedelta(days=(len(args.inpaths) - i - 1))
        dt = datetime.datetime.combine(day, datetime.time(9, 0))
        time_offsets.append(int(time.mktime(dt.timetuple())))
    remember(memory, args.inpaths, time_offsets, args.viddirs)
    
    if args.allow_recaption:
        agent.allow_recaption = True
    agent.set_memory(memory)
    
    while True:
        user_input = input("Enter command (or 'q' to quit): ")
        if user_input.strip().lower() == 'q':
            print("Exiting...")
            exit(0)
        
        result = agent.run(
            question=user_input,
            today=f"Today is {today.strftime('%A, %b %-d, %Y')}.",
            graph_type="full"
        )
        print("Result:", result)


if __name__ == "__main__":
    args = parse_args()
    assert len(args.inpaths) == len(args.viddirs), "The number of inpaths and viddirs must be equal."
    
    run(args)
    