import os
import copy
import traceback, sys
import base64
from io import BytesIO
from typing import Sequence
import json
import cv2
from PIL import Image as PILImage
from langchain_core.messages import ToolMessage
import numpy as np

from sensor_msgs.msg import Image

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
        
def ros_image_to_pil(ros_image):
    # Convert raw image data to numpy array
    np_arr = np.frombuffer(ros_image.data, dtype=np.uint8)

    # Reshape based on encoding
    if ros_image.encoding == "rgb8":
        image = np_arr.reshape((ros_image.height, ros_image.width, 3))
    elif ros_image.encoding == "mono8":
        image = np_arr.reshape((ros_image.height, ros_image.width))
    else:
        raise ValueError(f"Unsupported encoding: {ros_image.encoding}")

    return PILImage.fromarray(image)

def pil_to_utf8(pil_img: PILImage) -> str:
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")  # or "JPEG" etc.
    encoded_bytes = base64.b64encode(buffer.getvalue())
    utf8_str = encoded_bytes.decode("utf-8")
    return utf8_str

def numpy_to_ros_image(np_image, encoding="rgb8"):
    """ Convert a NumPy image (OpenCV) to a ROS Image message. """
    ros_image = Image()
    ros_image.height = np_image.shape[0]
    ros_image.width = np_image.shape[1]
    ros_image.encoding = encoding  # "bgr8" for OpenCV images, "rgb8" for PIL images
    ros_image.is_bigendian = 0
    ros_image.step = np_image.shape[1] * np_image.shape[2]  # width * channels
    ros_image.data = np_image.tobytes()
    return ros_image
        
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