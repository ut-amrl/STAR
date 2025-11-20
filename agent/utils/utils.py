import os
import ast
import copy
import traceback, sys
import base64
from io import BytesIO
from typing import Sequence
import json
import cv2
from typing import List, Dict, Optional, Any
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import numpy as np
import math
from typing import Callable
from datetime import datetime, timezone
import time
import uuid, tempfile
from pathlib import Path

# LangeChain imports
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

# ROS imports
from sensor_msgs.msg import Image

# Custom imports
from memory.memory import MilvusMemory, MemoryItem

class TempJsonStore:
    """
    Save short-lived JSON blobs to a temp dir with manual GC.
    - Atomic writes (write to .tmp then os.replace)
    - 0600 perms
    - GC by age (mtime)
    """
    def __init__(self, app_subdir: str = "agent_tmp_json"):
        # default under system /tmp, e.g. /tmp/agent_tmp_json
        self.root = Path(tempfile.gettempdir()) / app_subdir
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, file_id: str) -> Path:
        return self.root / f"{file_id}.json"

    def save(self, payload: Any, file_id: Optional[str] = None) -> dict:
        """
        Save JSON and return metadata with id & path.
        """
        file_id = file_id or uuid.uuid4().hex
        path = self._path_for(file_id)
        tmp = path.with_suffix(".json.tmp")

        # write atomically
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)

        # return {"id": file_id, "path": str(path), "uri": f"tmpjson://{file_id}"}
        return file_id

    def load(self, file_id_or_path: str) -> Any:
        """
        Load JSON (by id or full path).
        """
        path = Path(file_id_or_path)
        if path.suffix != ".json" or not path.exists():
            path = self._path_for(str(file_id_or_path).replace("tmpjson://", ""))
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete(self, file_id_or_path: str) -> bool:
        """
        Delete file if present. Returns True if deleted.
        """
        path = Path(file_id_or_path)
        if path.suffix != ".json" or not path.exists():
            path = self._path_for(str(file_id_or_path).replace("tmpjson://", ""))
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def gc(self, max_age_seconds: int = 3600) -> int:
        """
        Remove files older than `max_age_seconds`. Returns count removed.
        """
        now = time.time()
        removed = 0
        for p in self.root.glob("*.json"):
            try:
                if now - p.stat().st_mtime > max_age_seconds:
                    p.unlink()
                    removed += 1
            except FileNotFoundError:
                pass
        return removed

class SearchProposal:
    def __init__(self, 
                 summary: str, 
                 instance_description: str,
                 position: List[float], 
                 theta: float, 
                 records: List[dict]):
        self.summary: str = summary
        self.instance_description: str = instance_description  # Description of the object instance
        self.position: List[float] = position  # [x, y, z]
        self.theta: float = theta  # Orientation in radians
        self.records: List[dict] = records  # Original record from the database
        self.searched_poses: List[tuple] = []  # List of (position, theta) tuples for all searched poses
        
        self.reference_resolution_records: dict = None
        self.retrieval_grounding_records: dict = None
        
        self.instance_name: str = None
        self.has_picked: bool = False
        self.visible_instances: List[str] = []  # List of visible instances in the scene
        
    def __str__(self):
        pos_str = f"({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})"
        theta_deg = math.degrees(self.theta)
        return (f"SearchProposal: '{self.summary}' at position {pos_str}, "
                f"theta={theta_deg:.1f}°: {self.records}")
    
    def to_message(self):
        pass
    
    def get_viz_path(self):
        image_path_fn = lambda vidpath, frame: os.path.join(vidpath, f"{frame:06d}.png")
        
        record = self.records[0]
        vidpath = record["vidpath"]
        start_frame = record["start_frame"]
        end_frame = record["end_frame"]
        frame = (start_frame + end_frame) // 2
        
        return image_path_fn(vidpath, frame)
    
    def to_output(self):
        return {
            "instance_name": self.instance_name,
            "has_picked": self.has_picked,
            "summary": self.summary,
            "instance_description": self.instance_description,
            "position": self.position,
            "theta": self.theta,
            "records": self.records,
        }
    

class SearchInstance:
    def __init__(self, type: str = "memory"):
        self.type: str = type  # "mem" for memory, "world" for world
        self.inst_desc: str = ""
        self.inst_viz_path: str = None
        self.annotated_inst_viz = None
        self.annotated_bbox = None
        
        self.found: str = "unknown"  # "unknown", "yes", "no"
        self.past_observations: List[Dict] = []  # TODO likely needs to be refactored
        
    def __str__(self):
        return f"SearchInstance(inst_desc={self.inst_desc}, inst_viz_path={self.inst_viz_path}, found_in_{self.type}={self.found})"
    
    def to_message(self):
        message = []
        
        if self.type == "memory":
            txt_msg = [{"type": "text", "text": f"Current Memory Search Instance and its most update-to-update status: {self.__str__()}"}]
        else:
            txt_msg = [{"type": "text", "text": f"Current World Search Instance and its most update-to-update status: {self.__str__()}"}]
        
        message += txt_msg
        
        if self.inst_viz_path:
            try:
                img = PILImage.open(self.inst_viz_path).convert("RGB")  # RGBA to preserve color + alpha
            except Exception:
                # If the image cannot be opened, skip adding the image message
                return message
            
            # Draw the record ID with background
            draw = ImageDraw.Draw(img)
            text = f"Current Search Instance: {self.inst_desc}"
            font = ImageFont.load_default()
            text_size = draw.textbbox((0, 0), text, font=font)  # (left, top, right, bottom)
            padding = 4
            bg_rect = (
                text_size[0] - padding,
                text_size[1] - padding,
                text_size[2] + padding,
                text_size[3] + padding
            )
            draw.rectangle(bg_rect, fill=(0, 0, 0))  # Black background
            draw.text((text_size[0], text_size[1]), text, fill=(255, 255, 255), font=font)
            
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            img_msg = [get_vlm_img_message(img, type="gpt")]
            
            message += img_msg
        
        return message
    
class Task:
    def __init__(self, task_desc: str):
        self.task_desc: str = task_desc
        self.search_proposal: SearchProposal = None
        self.memory_search_instance: SearchInstance = None
        self.world_search_instance: SearchInstance = None
        
        self.searched_in_space: list = []
        self.searched_in_time: list = []


def get_viz_path(memory: MilvusMemory, record_id) -> str:
    record = memory.get_by_id(record_id, return_parsed=True)
    if record is None:
        return None

    image_path_fn = lambda vidpath, frame: os.path.join(vidpath, f"{frame:06d}.png")
        
    vidpath = record["vidpath"]
    start_frame = record["start_frame"]
    end_frame = record["end_frame"]
    frame = (start_frame + end_frame) // 2
        
    return image_path_fn(vidpath, frame)

def get_image_message_for_record(record_id: int, 
                                 viz_path: str, 
                                 tool_call_id: str = None,
                                 return_assistant_text: bool = True):
    try:
        img = PILImage.open(viz_path).convert("RGB")
    except Exception as e:
        return [{"type": "text", "text": f"Cannot obtain image for record {record_id}: {e}"}]
    
    # Draw the record ID with background
    draw = ImageDraw.Draw(img)
    text = f"Record: {record_id}"
    font = ImageFont.load_default()
    text_size = draw.textbbox((0, 0), text, font=font)  # (left, top, right, bottom)
    padding = 4
    bg_rect = (
        text_size[0] - padding,
        text_size[1] - padding,
        text_size[2] + padding,
        text_size[3] + padding
    )
    draw.rectangle(bg_rect, fill=(0, 0, 0))  # Black background
    draw.text((text_size[0], text_size[1]), text, fill=(255, 255, 255), font=font)
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    img_msg = [get_vlm_img_message(img, type="gpt")]
    
    if tool_call_id:
        txt = f"This is the image observation made at Record {record_id} from path {viz_path} (image for tool_call id {tool_call_id}):"
    else:
        txt = f"This is the image observation made at Record {record_id} from path {viz_path}:"
    txt_msg = [{"type": "text", "text": txt}]
    if return_assistant_text:
        return txt_msg + img_msg
    else:
        return img_msg

def is_image_inspection_result(content: str) -> bool:
    try:
        normalized = content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
        parsed = ast.literal_eval(normalized)

        if not isinstance(parsed, dict):
            return False

        # Ensure all keys are digits and values are .png strings
        for k, v in parsed.items():
            if not isinstance(k, str) or not k.isdigit():
                return False
            if not isinstance(v, str) or not v.lower().endswith(".png"):
                return False

        return True

    except Exception:
        return False
    
def is_recall_tool_result(content: str) -> bool:
    return is_recall_last_seen_result(content) or \
           is_recall_best_matches_result(content) or \
           is_recall_all_result(content)   
    
def is_recall_last_seen_result(content: str) -> bool:
    try:
        normalized = content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
        parsed = ast.literal_eval(normalized)

        if not isinstance(parsed, dict):
            return False

        # Check if 'tool_name' field exists and is 'recall_last_seen'
        return parsed.get("tool_name") == "recall_last_seen"

    except Exception:
        return False
    
def is_recall_best_matches_result(content: str) -> bool:
    try:
        normalized = content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
        parsed = ast.literal_eval(normalized)

        if not isinstance(parsed, dict):
            return False

        return parsed.get("tool_name") == "recall_best_matches"

    except Exception:
        return False
    
def is_recall_all_result(content: str) -> bool:
    try:
        normalized = content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
        parsed = ast.literal_eval(normalized)

        if not isinstance(parsed, dict):
            return False

        return parsed.get("tool_name") == "recall_all_matches"

    except Exception:
        return False
    
def is_search_in_time_terminate_result(msg) -> bool:
    try:
        if type(msg) == ToolMessage and hasattr(msg, "name") and msg.name == "terminate":
            return True
        return False

    except Exception:
        return False
    
def is_task_terminate_result(msg) -> bool:
    try:
        if type(msg) == ToolMessage and hasattr(msg, "name") and msg.name == "terminate_task":
            return True
        return False

    except Exception:
        return False
    
def has_file_id(content: str) -> bool:
    try:
        normalized = content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
        parsed = json.loads(normalized)

        if not isinstance(parsed, dict):
            return False

        return "file_id" in parsed

    except Exception:
        return False
    
def get_file_id(content: str) -> Optional[str]:
    """
    Extracts the file_id from the content if it exists.
    Returns None if no file_id is found.
    """
    try:
        normalized = content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
        parsed = json.loads(normalized)

        if not isinstance(parsed, dict):
            return None

        return parsed.get("file_id")

    except Exception:
        return ""
    
def get_file_id_messages(store: TempJsonStore, file_id: str):
    try:
        data = store.load(file_id)
        messages = [
            {"type": "text", "text": f"These are the observations stored in file_id {file_id}:"},
        ]
        messages += data.get("images", [])
        return messages
    except Exception as e:
        import pdb; pdb.set_trace()
        return None
        
def parse_and_pretty_print_tool_message(content: str) -> str:
    
    def is_memory_record_list(obj) -> bool:
        return (
            isinstance(obj, list)
            and all(isinstance(entry, dict) for entry in obj)
            and all(key in obj[0] for key in ("id", "timestamp", "position", "theta", "text"))
        )
        
    def get_pretty_memory_record_list_str(data: List[dict]) -> str:
        lines = []
        for d in data:
            timestamp = float(d["timestamp"])
            readable_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            pos = eval(d["position"]) if isinstance(d["position"], str) else d["position"]
            pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            theta = d["theta"]
            text = d["text"]

            lines.append(
                f"• [Record {d['id']}] It was {readable_time}, and I was at position {pos_str} "
                f"and angle θ = {theta:.2f}. I saw the following: {text}."
            )
        return "\n".join(lines)
    
    try:
        # Some messages are double-braced {{}} for safety → fix before parsing
        data = ast.literal_eval(content)
        
        if is_memory_record_list(data):
            record_list = data
            # Sort chronologically
            record_list = sorted(record_list, key=lambda d: float(d["timestamp"]))
            final_output = get_pretty_memory_record_list_str(record_list)
            return final_output.replace("{", "{{").replace("}", "}}")
        
        elif isinstance(data, dict) and "tool_name" in data and "records" in data and "summary" in data:
            summary = data["summary"]
            summary_output = f"{summary} This is all the memory records I found for you:\n"
            
            tool_name = data["tool_name"]
            record_list = data["records"]
            if tool_name == "recall_all_terminate":
                record_list = sorted(record_list, key=lambda d: float(d["timestamp"]))
            record_list_output = get_pretty_memory_record_list_str(record_list)
            
            final_output = summary_output + record_list_output
            return final_output.replace("{", "{{").replace("}", "}}")
        
        return content

    except Exception as e:
        return content.replace("{", "{{").replace("}", "}}")  # Fallback if not parsable

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size 

    def find(self, x):
        if self.parent[x] != x:
            # Path compression
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Find roots
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in the same set

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True  # Successfully merged

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_groups(self):
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            groups[root].append(i)
        return list(groups.values())

def downsample_consecutive_ids(ids, rate):
    if not ids:
        return []

    result = []
    chunk = [ids[0]]

    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1] + 1:
            chunk.append(ids[i])
        else:
            # Process the current chunk
            result.extend(chunk[::rate])
            chunk = [ids[i]]

    # Don't forget the last chunk
    result.extend(chunk[::rate])
    return result

def last_multi_group_index(grouped_ids):
    for i, group in enumerate(grouped_ids):
        if len(group) == 1:
            return i - 1 if i > 0 else -1
    return len(grouped_ids) - 1  # All groups have size > 1

def angle_diff(a0: float, a1: float) -> float:
    """Compute minimal signed difference between two angles in radians."""
    two_pi = 2 * math.pi
    angle = a0 - a1
    angle -= two_pi * round(angle / two_pi)
    return angle

def file_to_string(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read().strip()
    
def try_except_continue(state, func):
    from openai import BadRequestError
    while True:
        try:
            ret = func(state)
            return ret
        except BadRequestError as e:
            import pdb; pdb.set_trace()
            print("[ERROR] OpenAI API BadRequestError:", e)
        except Exception as e:
            print("I crashed trying to run:", func)
            print("Here is my error")
            print(e)
            traceback.print_exception(*sys.exc_info())
        
def opencv_to_ros_image(np_image):
    ros_image = Image()
    ros_image.height = np_image.shape[0]
    ros_image.width = np_image.shape[1]
    ros_image.encoding = "bgr8"
    ros_image.is_bigendian = 0
    ros_image.step = np_image.shape[1] * np_image.shape[2]  # width * channels
    ros_image.data = np_image.tobytes()
    return ros_image

def pil_to_ros_image(pil_image):
    if pil_image.mode != "RGB":
        raise ValueError(f"PIL image must be in RGB mode, got {pil_image.mode}")
    np_image = np.array(pil_image)

    ros_image = Image()
    ros_image.height = np_image.shape[0]
    ros_image.width = np_image.shape[1]
    ros_image.encoding = "rgb8"
    ros_image.is_bigendian = 0
    ros_image.step = np_image.shape[1] * np_image.shape[2]  # width * channels
    ros_image.data = np_image.tobytes()
    return ros_image
        
def ros_image_to_pil(ros_image):
    # Convert raw image data to numpy array
    np_arr = np.frombuffer(ros_image.data, dtype=np.uint8)

    # Reshape based on encoding
    if ros_image.encoding == "rgb8" or ros_image.encoding == "bgr8":
        image = np_arr.reshape((ros_image.height, ros_image.width, 3))
        if ros_image.encoding == "bgr8":
            image = image[..., ::-1]  # Convert BGR to RGB for PIL
    elif ros_image.encoding == "mono8":
        image = np_arr.reshape((ros_image.height, ros_image.width))
    else:
        raise ValueError(f"Unsupported encoding: {ros_image.encoding}")

    return PILImage.fromarray(image)

def ros_image_to_vlm_message(ros_image):
    pil_img = ros_image_to_pil(ros_image)
    utf8_img = pil_to_utf8(pil_img)
    return [get_vlm_img_message(utf8_img)]

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

def get_image(
    vidpath: str, 
    start_frame: str, 
    end_frame: str, 
    type: str = "opencv", 
    resize: bool = False,
    image_path_fn: Callable[[str, int], str] = None
):
    if image_path_fn is None:
        image_path_fn = lambda vidpath, frame: os.path.join(vidpath, f"{frame:06d}.png")
    
    start_frame, end_frame = int(start_frame), int(end_frame)
    frame = (start_frame + end_frame) // 2
    imgpath = image_path_fn(vidpath, frame)
    
    if type.lower() == "opencv":
        img = cv2.imread(imgpath)
    elif type.lower() == "pil":
        img = PILImage.open(imgpath)
    elif type.lower() == "utf-8":
        if resize:
            img = PILImage.open(imgpath).convert("RGB")  # RGBA to preserve color + alpha
            img = img.resize((512, 512), PILImage.BILINEAR)
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img = base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            with open(imgpath, "rb") as imgfile:
                data = imgfile.read()
                img = base64.b64encode(data)
                img = copy.copy(img.decode("utf-8"))
    else:
        ValueError("Invalid image data type: only support opencv, PIL, or utf-8")
    return img

def get_image_from_path(imgpath: str, type: str):
    if type.lower() == "opencv":
        img = cv2.imread(imgpath)
    elif type.lower() == "pil":
        img = PILImage.open(imgpath)
    elif type.lower() == "utf-8":
        with open(imgpath, "rb") as imgfile:
            data = imgfile.read()
            img = base64.b64encode(data)
            img = copy.copy(img.decode("utf-8"))
    elif type.lower() == "ros":
        openvc_img = cv2.imread(imgpath)
        img = opencv_to_ros_image(openvc_img)
    else:
        raise ValueError("Invalid image data type: only support opencv, PIL, utf-8, or ros") 
    return img

def get_image_from_record(
    record: dict, 
    type: str = "opencv", 
    resize: bool = False,
    image_path_fn: Callable[[str, int], str] = None
):
    return get_image(
        record["vidpath"], 
        record["start_frame"], 
        record["end_frame"], 
        type=type, 
        resize=resize,
        image_path_fn=image_path_fn
    )

def get_images(
    vidpath: str, 
    start_frame: str, 
    end_frame: str, 
    type: str = "opencv", 
    step: int = 1, 
    image_path_fn: Callable[[str, int], str] = None
):
    if image_path_fn is None:
        image_path_fn = lambda vidpath, frame: os.path.join(vidpath, f"{frame:06d}.png")
    
    images = []
    for frame in range(start_frame, end_frame, step):
        imgpath = image_path_fn(vidpath, frame)
        if type.lower() == "opencv":
            img = cv2.imread(imgpath)
        elif type.lower() == "pil":
            img = PILImage.open(imgpath)
        elif type.lower() == "utf-8":
            with open(imgpath, "rb") as imgfile:
                data = imgfile.read()
                img = base64.b64encode(data).decode("utf-8")
        else:
            ValueError("Invalid image data type: only support opencv, PIL, or utf-8")
        images.append(img)
    return images

def get_images_from_record(
    record: dict, 
    type: str = "opencv", 
    step: int = 1, 
    image_path_fn: Callable[[str, int], str] = None
):
    return get_images(record["vidpath"], record["start_frame"], record["end_frame"], type=type, step=step, image_path_fn=image_path_fn)

def debug_vid(vid, debugdir: str):
    os.makedirs(debugdir, exist_ok=True)
    for filename in os.listdir(debugdir):
            filepath = os.path.join(debugdir, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
    images = get_images(vid["vidpath"], vid["start_frame"], vid["end_frame"])
    for i, img in enumerate(images):
            imgpath = os.path.join(debugdir, f"{i:06d}.png")
            import cv2; cv2.imwrite(imgpath, img)

def get_vlm_img_message(img, type: str = ""):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
        
def replace_messages(current: Sequence, new: Sequence):
    """Custom update strategy to replace the previous value with the new one."""
    return new

def filter_retrieved_record(messages: list):
    tool_responses = [msg.content for msg in filter(lambda x: isinstance(x, ToolMessage), messages)]
    records = []
    try:
        for response in tool_responses:
            for r in json.loads(response):
                records.append(r)
        unique_by_ids = {r["id"]: r for r in records}
        records = sorted(unique_by_ids.values(), key=lambda x: x["id"])
    except:
        import pdb; pdb.set_trace()
        print()
    return records
            
def parse_db_records_for_llm(messages):
    is_str = False
    if type(messages) is not list:
        is_str = True
        
    if is_str:
        messages = [messages]
    
    processed_records = []
    for record in messages:
        # Parse position
        pos = eval(record['position'])
        # Format timestamp
        dt = datetime.fromtimestamp(record['timestamp'])
        dt_str = dt.strftime("%y-%m-%d, %H:%M:%S")
        # Compose text
        text = f"I was at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) at {dt_str}. {record['text']}"
        processed_records.append({
            "record_id": record["id"],
            "text": text
        })

    parsed_processed_records = [json.dumps(record) for record in processed_records]
    parsed_processed_records = [
        f"[RETRIEVED OBSERVATION] {record}" for record in parsed_processed_records
    ]
    if is_str:
        return parsed_processed_records[0]
    else:
        return parsed_processed_records


### LLM/VLM/ML tools
def get_depth(xyxy, depth, padding:int=8):
    xyxy = [int(x) for x in xyxy]
    # If object is too small, return invalid depth
    if xyxy[2] - xyxy[0] < 4 or xyxy[3] - xyxy[1] < 4:
        return np.nan
    center_x = (xyxy[0] + xyxy[2]) // 2
    center_y = (xyxy[1] + xyxy[3]) // 2
    xl = max(xyxy[0]+2, center_x-padding)
    xr = min(xyxy[2]-2, center_x+padding)
    yl = max(xyxy[1]+2, center_y-padding)
    yr = min(xyxy[3]-2, center_y+padding)
    
    Zs = depth[yl:yr+1, xl:xr+1]
    mask = ~np.isnan(Zs)
    Z = Zs[mask]
    
    if len(Z) > 2:
        mean, std = np.mean(Z), np.std(Z)
        threshold = 2
        Z = Z[np.fabs(Z - mean) <= threshold * std]
        
    if len(Z) == 0:
        return np.nan
    print("depth: ", Z.mean())
    return Z.mean()

def is_txt_instance_observed(query_img, query_txt: str, depth = None, logger = None):
    response = request_bbox_detection_service(query_img, query_txt)
    for detection in response.bounding_boxes.bboxes:
        z = get_depth(detection.xyxy, depth)
        if (z is not np.nan) and (z < 2.5):
            if logger:
                logger.info(f"Observed instannce {query_txt} {z:.2f} m away.")
            return True
        if logger:
            logger.info(f"Observed instannce {query_txt} {z:.2f} m away (Too far).")
    if logger:
        logger.info(f"Failed to observe instannce {query_txt}")
    return False

def is_viz_instance_observed(vlm, vlm_processor, obs, instance):
    messages = [
        {
            "role": "system",
            "content": "You will be provided with an image and a text description of an instance. Your job is to determine if the instance appears on the foreground of the image. Please provide a 'yes' or 'no' answer to the following question. Do not answer anything other than 'yes' or 'no'"
        },
        {
            "role": "user",
            "content": [
                obs,
                {
                    "type": "text", 
                    "text": f"Does this image contain the following instance: {instance}?"
                },
            ]
        }
    ]
    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(vlm.device)
    generated_ids = vlm.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return "yes" in output_text[0].lower()


## ROS Service Calls
import rospy
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import (
    GetImageSrv,
    GetImageSrvRequest,
    GetImageAtPoseSrv, 
    GetImageAtPoseSrvRequest, 
    SemanticObjectDetectionSrv, 
    SemanticObjectDetectionSrvRequest,
    PickObjectSrv,
    PickObjectSrvRequest,
)

def request_bbox_detection_service(ros_image, query_text: str):
    rospy.wait_for_service('grounding_dino_bbox_detector')
    try: 
        grounding_dino = rospy.ServiceProxy('grounding_dino_bbox_detector', SemanticObjectDetectionSrv)
        request = SemanticObjectDetectionSrvRequest()
        request.query_text = query_text
        request.query_image = ros_image
        response = grounding_dino(request)
        return response
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        
def request_pick_service(query_image: Image = Image(), query_txt: str = ""):
    rospy.wait_for_service("/Cobot/Pick")
    try: 
        pick_object = rospy.ServiceProxy("/Cobot/Pick", PickObjectSrv)
        request = PickObjectSrvRequest()
        request.query_image = query_image
        request.query_text = query_txt
        response = pick_object(request)
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    return response

def request_get_image_at_pose_service(goal_x:float, goal_y:float, goal_theta:float, logger = None):
    response = None
    if logger:
        logger.info(f"Requesting to navgiate to ({goal_x:.2f}, {goal_y:.2f}, {goal_theta:.2f})")
    rospy.wait_for_service("/Cobot/GetImageAtPose")
    try: 
        get_image_at_pose = rospy.ServiceProxy("/Cobot/GetImageAtPose", GetImageAtPoseSrv)
        request = GetImageAtPoseSrvRequest()
        request.x = goal_x
        request.y = goal_y
        request.theta = goal_theta
        response = get_image_at_pose(request)
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    if logger:
        if response and response.success:
            logger.info(f"Successfully navgiate to ({goal_x:.2f}, {goal_y:.2f}, {goal_theta:.2f})")
        else:
            logger.info(f"Failed to navgiate to ({goal_x:.2f}, {goal_y:.2f}, {goal_theta:.2f})")
    return response

def ask_chatgpt(model, prompt: str, images, question: str):
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    
    content_blocks = images + [{"type": "text", "text": question}]
    messages = [
        AIMessage(content=prompt),
        HumanMessage(content=content_blocks),
    ]
    
    response = model.invoke(messages)
    return response

def recaption(memory: MilvusMemory, task:str, record: dict, caption_fn):
    caption = caption_fn(task, record)
    caption_embedding = memory.embedder.embed_query(caption)  # Ensure caption is embedded before inserting
    position = record["position"]
    if type(position) == str:
        position = eval(position)
    
    item = MemoryItem(
        caption=caption,
        text_embedding=caption_embedding,
        time=record["timestamp"],
        position=position,
        theta=record["theta"],
        vidpath=record["vidpath"],
        start_frame=record["start_frame"],
        end_frame=record["end_frame"]
    )
    # Memory should handle caption embedding internally
    memory.insert(item)
    
    return caption

def caption_gpt(task: str, record: dict, model: ChatOpenAI = None, image_path_fn: Callable[[str, int], str] = None):
    if model is None:
        model = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))
    if image_path_fn is None:
        image_path_fn = lambda vidpath, frame: os.path.join(vidpath, f"{frame:06d}.png")
        
    images = get_images_from_record(
        record, type="utf-8", image_path_fn=image_path_fn
    )
    image_messages = []
    for image in images:
        image_message = get_vlm_img_message(image, type="gpt")
        image_messages.append(image_message)
    
    prompt = (
        "You are a robot assistant re-examining a previously observed scene to improve its memory. "
        "You are given the original caption and a user task. Your job is to revise the caption in light of the user's task, "
        "while preserving all general-purpose information.\n\n"

        "Instructions:\n"
        "- Do not remove or rewrite any part of the original caption that is unrelated to the task.\n"
        "- If the scene contains information relevant to the user's task, update the caption to include those details—both confirming and negating.\n"
        "- Use common sense reasoning to infer likely intent, usage, or relevance based on the task and the scene.\n"
        "- Always stay faithful to the visual input. If something appears incorrect or missing in the original caption, correct it based on the actual scene.\n"
        "- The result must be a single, standalone paragraph that is descriptive, detailed, and useful for both general memory and task resolution.\n"
        "- Do not mention the task explicitly. Incorporate task-relevant information naturally into the description.\n\n"

        "Your goal is to help the robot better remember and interpret this scene for the current task and future use."
        "Do not introduce yourself or comment on the video — just describe what is present and what is happening, "
        "with high factual precision that would help a robot remember and act on this scene. Your should respond in a single paragraph."
    )
    old_caption = record["text"]
    question = f"This is the original caption of the video: {old_caption}.\nUser task: {task}.\nCould you recaption the video in light of the user task?"
    response = ask_chatgpt(model, prompt, image_messages, question)
    return response.content

import json
import re

def parse_json(text: str):
    # 1. Extract from ```json ... ``` if present
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text.strip()
    
    # 2. Try parsing with json.loads
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 3. Fallback to eval (use with caution — safe only if you trust the input)
    try:
        return eval(json_str, {"__builtins__": {}})
    except Exception as e:
        raise ValueError(f"Could not parse response as JSON or Python dict: {e}")
    
def safe_gpt_invoke(flex_model, nonflex_model, *args, **kwargs):
    """
    Try flex_model.invoke(...) up to 3 times on transient errors (429, 5xx, timeouts, connection resets).
    Backoff: 1s, 2s, 4s. Then fall back to nonflex_model.invoke(...).

    Works for both:
      - safe_gpt_invoke(vlm_flex, vlm, messages)
      - safe_gpt_invoke(vlm_flex, vlm, chat_history=..., fact_prompt=..., question=...)
    """
    delays = [1, 2, 4]
    transient_markers = [
        "no healthy upstream",
        "temporarily unavailable",
        "upstream connect error",
        "connection reset",
        "bad gateway",
        "gateway timeout",
        "timed out",
        "timeout",
        "server overloaded",
        "retry later",
        "status code: 429",
        "rate limit",
    ]

    def is_transient(e: Exception) -> bool:
        code = getattr(e, "status_code", None) or getattr(e, "code", None)
        if isinstance(code, int) and (code == 429 or 500 <= code <= 599 or code in (408,)):
            return True
        msg = str(e).lower()
        return any(marker in msg for marker in transient_markers)

    # Try flex model with retries
    for attempt, delay in enumerate(delays, start=1):
        try:
            return flex_model.invoke(*args, **kwargs)
        except Exception as e:
            if is_transient(e):
                if attempt < len(delays):
                    print(f"[WARN] Transient error from flex (attempt {attempt}/3): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"[INFO] Flex still failing after {attempt} attempt(s): {e}. Falling back to non-flex.")
                    break
            raise  # non-transient → bubble up

    # Fallback to non-flex
    print("[INFO] Switching to non-flex model...")
    return nonflex_model.invoke(*args, **kwargs)