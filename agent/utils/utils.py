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
    if ros_image.encoding == "rgb8" or ros_image.encoding == "bgr8":
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
        
def get_image(vidpath: str, start_frame: str, end_frame: str, type: str = "opencv"):
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

def get_images(vidpath: str, start_frame: str, end_frame: str, type: str = "opencv"):
    images = []
    for frame in range(start_frame, end_frame+1):
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
        images.append(img)
    return images

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
        logger.info(f"Observed instannce {query_txt} {z:.2f} m away (Too far).")
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