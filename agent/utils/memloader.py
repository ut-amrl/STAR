import os, sys
import json
from time import strftime, localtime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from memory.memory import MilvusMemory
from memory.memory import MemoryItem

def remember(memory: MilvusMemory, inpaths: list, time_offsets: list, viddirs: list):
    for inpath, time_offset, viddir in zip(inpaths, time_offsets, viddirs):
        with open(inpath, 'r') as f:
            for entry in json.load(f):
                t, pos, caption, text_embedding, start_frame, end_frame = entry["time"], entry["base_position"], entry["base_caption"], entry["base_caption_embedding"], entry["start_frame"], entry["end_frame"]
                t += time_offset
                start_frame, end_frame = int(start_frame), int(end_frame)
                memory_item = MemoryItem(
                    caption=caption,
                    text_embedding=text_embedding,
                    time=t,
                    position=pos,
                    theta=0,
                    vidpath=viddir,
                    start_frame=start_frame,
                    end_frame=end_frame
                )
                memory.insert(memory_item)
                

def update_from_paths(memory: MilvusMemory, inpaths: list):
    for i, inpath in enumerate(inpaths):
        with open(inpath, 'r') as f:
            for data in [json.loads(line) for line in f]:
                memory_item = MemoryItem(
                    caption=data["caption"],
                    time=data["time"],
                    position=data["position"],
                    theta=data["theta"],
                    vidpath=data["vidpath"],
                    start_frame=data["start_frame"],
                    end_frame=data["end_frame"],
                )
                memory.insert(memory_item)
            

def remember_from_paths(memory: MilvusMemory, inpaths: list, time_offset: float, viddir: str):
    for i, inpath in enumerate(inpaths):
        start_time = None
        with open(inpath, 'r') as f:
            for entry in json.load(f):
                t, pos, caption, start_frame, end_frame = entry["time"], entry["base_position"], entry["base_caption"], entry["start_frame"], entry["end_frame"]
                
                # handle time
                if start_time is None:
                    start_time = t
                t = t - start_time + 1.5 + time_offset
                
                # handle pos
                if len(pos) == 2:
                    pos += [0.0]
                pos = [round(p, 1) for p in pos]
                
                timestr = strftime('%Y-%m-%d %H:%M:%S', localtime(t))
                capstr = f"At time={timestr}, the robot was at an average position of {pos}. "
                capstr += f"The robot saw the following: {caption}"
                
                # handle video path
                vidpath = viddir
                if "cobot" in inpath:
                    vidpath = os.path.join(vidpath, "bag")
                vidpath = os.path.join(vidpath, Path(inpath).stem)
                
                # insertion
                memory_item = MemoryItem(
                    caption=capstr,
                    time=t,
                    position=pos,
                    theta=0,
                    vidpath=vidpath,
                    start_frame=start_frame,
                    end_frame=end_frame
                )
                memory.insert(memory_item)
                
        time_offset += 86400 # offset 1 day
        