import os
import copy
import traceback, sys
import base64
from io import BytesIO
from typing import Sequence
import json
import cv2
from PIL import Image as PILImage
import numpy as np
import math
from typing import Callable

# LangeChain imports
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

# ROS imports
from sensor_msgs.msg import Image

# Custom imports
from memory.memory import MilvusMemory, MemoryItem

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
    for frame in range(start_frame, end_frame+1, step):
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
            
def parse_db_records_for_llm(messages: list):
    processed_records = [{"record_id": record["id"], "text": record["text"]} for record in messages]
    parsed_processed_records = [json.dumps(record) for record in processed_records]
    parsed_processed_records = [
        f"[RETRIEVED OBSERVATION] {record}" for record in parsed_processed_records
    ]
    return parsed_processed_records


### LLM/VLM/ML tools
from qwen_vl_utils import process_vision_info

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

def ask_qwen(vlm_model, vlm_processor, prompt: str, image, question: str):
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                image,
                {"type": "text", "text": question},
            ],
        },
    ]

    # Process inputs
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(vlm_model.device)
    
    generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

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