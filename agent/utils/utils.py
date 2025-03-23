import os
import copy
import traceback, sys
import base64
from typing import Sequence
import json
import cv2
from PIL import Image as PILImage
from langchain_core.messages import ToolMessage

def file_to_string(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read().strip()
    
def try_except_continue(state, func):
    while True:
        try:
            ret = func(state)
            return ret
        except Exception as e:
            print("I crashed trying to run:", func)
            print("Here is my error")
            print(e)
            traceback.print_exception(*sys.exc_info())
            continue
        
def get_image(vidpath: str, start_fram, end_frame, type: str = "opencv"):
    start_frame, end_frame = int(start_frame), int(end_frame)
    frame = (start_frame + end_frame) // 2
    imgpath = os.path.join(vidpath, f"{frame:06d}.png")
    if type.lower() == "opencv":
        img = cv2.imread(imgpath)
    elif type.lower() == "pil":
        img = PILImage.open(imgpath)
    elif type.lower() == "utf-8":
        with open(imgpath, "rb") as imgfile:
            data = imgfile.read()
            data = base64.b64encode(data)
            img = copy.copy(img.decode("utf-8"))
    else:
        ValueError("Invalid image data type: only support opencv, PIL, or utf-8")
    return img

def get_vlm_img_message(img, type: str = "qwen"):
    if "qwen" in type:
        return {"type": "image", "image": f"data:image/png;base64,{img}"}
    else:
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
        
def replace_messages(current: Sequence, new: Sequence):
    """Custom update strategy to replace the previous value with the new one."""
    return new

def filter_retrieved_record(messages: list):
    tool_responses = [msg.content for msg in filter(lambda x: isinstance(x, ToolMessage), messages)]
    records = []
    for response in tool_responses:
        for r in json.loads(response):
            records.append(r)
    unique_by_ids = {r["id"]: r for r in records}
    records = sorted(unique_by_ids.values(), key=lambda x: x["id"])
    return records
            
def parse_db_records_for_llm(messages: list):
    processed_records = [{"record_id": record["id"], "text": record["text"]} for record in messages]
    parsed_processed_records = [json.dumps(record) for record in processed_records]
    return parsed_processed_records