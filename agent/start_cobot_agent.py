import argparse
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import torch
import re
import os
import threading
import time
import numpy as np
import json

from agent import Agent
from utils.utils import (
    ros_image_to_pil,
    request_get_image_at_pose_service,
    get_vlm_img_message,
    ask_qwen,
    ask_chatgpt
)
from utils.memloader import remember_from_paths, update_from_paths
from utils.function_wrapper import FunctionsWrapper
from memory.memory import MilvusMemory, MemoryItem

import rospy
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.msg import BBox2DMsg, BBox2DArrayMsg
from amrl_msgs.srv import RememberSrv, SemanticObjectDetectionSrv, SemanticObjectDetectionSrvResponse


OBJECT_DETECTOR = None
CAPTIONER = None
VLM_MODEL, VLM_PROCESSOR = None, None
GPT = None
MEMORY = None
SAVEPATH = None
OBS_SAVEPATH = None
FILE = None

gd_device = "cuda:2"

def parse_args():
    default_query = "<video>\n You are a wandering around a household area. Please describe in detail what you see in the few seconds of the video. \
        Focus on objects. You can also descripe events/activities, people and their actions, and other notable details. \
        Provide enough detail about objects (e.g., colors, patterns, logos, or states) to ensure they can be identified through text alone. For example, instead of just 'a box,' describe its color, any images or logos on it, and any distinguishing marks. \
        Think step by step about these details and be very specific. \
        Describe the video directly without any introductory phrases or extra commentary."
    parser = argparse.ArgumentParser()
    # VILA
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--bagname", type=str, default="")
    parser.add_argument("--captioner_name", type=str, default="VILA1.5-8b")
    parser.add_argument("--seconds_per_caption", type=int, default=3)
    parser.add_argument("--num-video-frames", type=int, default=5)
    parser.add_argument("--query", type=str, default=default_query)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--obs_savepath", type=str, required=True)
    
    # Grounding DINO
    parser.add_argument("--gd_config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--gd_checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )

    parser.add_argument("--gd_box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--gd_text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--gd_token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug mode, default=False")
    
    parser.add_argument("--case", type=str)
    
    args = parser.parse_args()
    return args

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = PILImage.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def load_grounding_dino_model(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)
    args.device = gd_device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, device="cuda", token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    labels, confs = [], []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        labels.append(pred_phrase)
        confs.append(logit.max().item())

    return boxes_filt, pred_phrases, labels, confs

def handle_object_detection_request(req):
    # rospy.loginfo(f"Received request with query: {req.query_text}")
    
    query_pil_img = ros_image_to_pil(req.query_image)
    query_txt = req.query_text
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    query_image, _ = transform(query_pil_img, None) 
    # run model
    boxes_filt, pred_phrases, labels, confs = get_grounding_output(
        OBJECT_DETECTOR, query_image, query_txt, GD_BOX_THRESHOLD, GD_TEXT_THRESHOLD, device=gd_device, token_spans=eval(f"{GD_TOKEN_SPANS}")
    )
    
    if VERBOSE:
        size = query_pil_img.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        query_pil_img.save(os.path.join("debug", "raw_image.jpg"))
        image_with_box = plot_boxes_to_image(query_pil_img, pred_dict)[0]
        image_with_box.save(os.path.join("debug", "pred.jpg"))
    
    H, W = query_pil_img.size
    bbox_arr_msg = BBox2DArrayMsg(header=req.query_image.header)
    for box, label, conf in zip(boxes_filt, labels, confs):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        xyxy = box
        bbox_arr_msg.bboxes.append(BBox2DMsg(label=label, conf=conf, xyxy=xyxy))
    rospy.loginfo(f"Sending results with query: {req.query_text}")
    return SemanticObjectDetectionSrvResponse(bounding_boxes=bbox_arr_msg)

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

class VILACaptioner:
    def __init__(self, args):
        # Model
        disable_torch_init()

        self.model_name = get_model_name_from_path(args.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, self.model_name, args.model_base)

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        self.args = args

    def caption(self, images: list[PILImage.Image], prompt: str):
        args = self.args
        # Model
        disable_torch_init()

        qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                # print("no <image> tag found in input. Automatically append one at the beginning of text.")
                # do not repeatively append the prompt.
                if self.model.config.mm_use_im_start_end:
                    qs = (image_token_se + "\n") * len(images) + qs
                else:
                    qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
            
        images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[
                    images_tensor,
                ],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs
    
def handle_observation_request(req, use_vila: bool = False, use_qwen: bool = False, use_gpt: bool = True):
    timestamp = req.timestamp
    position = [req.x, req.y, req.theta]
    theta = req.theta
    
    pil_images = []
    for imgmsg in req.video:
        pil_image = ros_image_to_pil(imgmsg)
        pil_images.append(pil_image)
        
    captions = []
    if use_vila:
        caption = CAPTIONER.caption(pil_images, PROMPT)
        captions.append(caption)
    
    if use_qwen or use_gpt:
        import io; import base64
        buffer = io.BytesIO()
        pil_images[0].save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
    if use_qwen:
        img = get_vlm_img_message(img_base64)
        caption = ask_qwen(VLM_MODEL, 
                            VLM_PROCESSOR, 
                            "Caption the same image. Make sure you capture all objects in great details. Respond in a single paragraph.",
                            img, 
                            "What do you see in the image?")
        captions.append(caption)
        
    if use_gpt:
        img = get_vlm_img_message(img_base64, type="chat-gpt")
        caption = ask_chatgpt(GPT, 
                             "Caption the same image. Make sure you capture all objects in great details. Respond in a single paragraph.",
                             [img], 
                             "What do you see in the image?")
        captions.append(caption)
    
    caption = "\n".join(captions)
    
    filenames = sorted(os.listdir(OBS_SAVEPATH))
    frame = 0
    if len(filenames) != 0:
        frame = int(filenames[-1][:-4])
    start_frame = frame + 1
    end_frame = frame + len(pil_images)
    
    item = MemoryItem(
        caption=caption,
        time=timestamp,
        position=position,
        theta=theta,
        vidpath=OBS_SAVEPATH,
        start_frame=start_frame,
        end_frame=end_frame
    )
    MEMORY.insert(item, images=pil_images)
    
    data = {
        "time": timestamp,
        "caption": caption,
        "position": position,
        "theta": theta,
        "vidpath": OBS_SAVEPATH,
        "start_frame": start_frame,
        "end_frame": end_frame
    }
    FILE.write(json.dumps(data) + "\n")
    FILE.flush()
    
    return True

def start_services():
    rospy.Service('/memory/observe', RememberSrv, handle_observation_request)
    rospy.loginfo("Memory observation service is ready.")
    # rospy.Service('grounding_dino_bbox_detector', SemanticObjectDetectionSrv, handle_object_detection_request)
    # rospy.loginfo("GroundingDINO service is ready.")
    rospy.spin()

if __name__ == "__main__":
    args = parse_args()
    VERBOSE = args.verbose
    
    if not args.replay:
        # captioner
        SAVEPATH = args.obs_savepath; 
        os.makedirs(SAVEPATH, exist_ok=True)
        from datetime import datetime
        SAVEPATH = os.path.join(SAVEPATH, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(SAVEPATH, exist_ok=True)
        
        OBS_SAVEPATH = os.path.join(SAVEPATH, "images")
        os.makedirs(OBS_SAVEPATH, exist_ok=True)
        filepath = os.path.join(SAVEPATH, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
        FILE = open(filepath, "a")
        
        from langchain_openai import ChatOpenAI
        GPT = ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
        
        PROMPT = args.query
        # CAPTIONER = VILACaptioner(args)
    
    # from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    # from qwen_vl_utils import process_vision_info
    # VLM_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map={"": 1}
    # )
    # VLM_PROCESSOR = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    if args.case:
        if args.case == "case1":
            # Bring me a cup (last seen)
            inpaths = [
                "/robodata/taijing/RobotMem/data/captions/cobot/2025-04-05-17-24-49_VILA1.5-8b_3_secs.json",
            ]
            MEMORY = MilvusMemory("test2", obs_savepth=OBS_SAVEPATH, db_ip='127.0.0.1')
            MEMORY.reset()
            t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400
            remember_from_paths(MEMORY, inpaths, t_offset, viddir="/robodata/taijing/RobotMem/data/images")
            task = "Bring me a cup."
            # debug/agent/2025-04-05/2025-04-05_18-45-27.log
            # [8.3, 54.3, 3.1]
            
        elif args.case == "case2":
            # TODO
            inpaths = [
                "/robodata/taijing/RobotMem/data/captions/cobot/2025-04-05-17-07-31_VILA1.5-8b_3_secs.json",
                "/robodata/taijing/RobotMem/data/captions/cobot/2025-04-05-17-41-23_VILA1.5-8b_3_secs.json"
            ]
            MEMORY = MilvusMemory("test2", obs_savepth=OBS_SAVEPATH, db_ip='127.0.0.1')
            MEMORY.reset()
            t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400
            remember_from_paths(MEMORY, inpaths, t_offset, viddir="/robodata/taijing/RobotMem/data/images")
            task = "Today is Feb 08, 2025. Bring me the cup that was on the table yesterday."
            
        elif args.case == "case3":
            inpaths = [
                "/robodata/taijing/RobotMem/data/captions/cobot/2025-04-05-17-24-49_VILA1.5-8b_3_secs.json",
                "/robodata/taijing/RobotMem/data/captions/cobot/2025-04-05-17-41-23_VILA1.5-8b_3_secs.json"
            ]
            MEMORY = MilvusMemory("test2", obs_savepth=OBS_SAVEPATH, db_ip='127.0.0.1')
            MEMORY.reset()
            t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400
            remember_from_paths(MEMORY, inpaths, t_offset, viddir="/robodata/taijing/RobotMem/data/images")
            task = "Today is Feb 07, 2025. Bring me the cup that was next to the plate yesterday."
            # debug/agent/2025-04-05/2025-04-05_18-35-29.log
            # [10.4, 60.2, 2.8]
            
        elif args.case == "case4":
            inpaths = [
                "/robodata/taijing/RobotMem/data/captions/cobot/2025-04-05-17-07-31_VILA1.5-8b_3_secs.json",
                "/robodata/taijing/RobotMem/data/captions/cobot/2025-04-05-17-41-23_VILA1.5-8b_3_secs.json"
            ]
            MEMORY = MilvusMemory("test2", obs_savepth=OBS_SAVEPATH, db_ip='127.0.0.1')
            MEMORY.reset()
            t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400
            remember_from_paths(MEMORY, inpaths, t_offset, viddir="/robodata/taijing/RobotMem/data/images")
            task = "Today is Feb 07, 2025. Bring me the object that was on the shelf yesterday."
            # debug/agent/2025-04-05/2025-04-05_18-39-23.log
            # [10.4, 60.2, 2.8]
            
        
        agent = Agent()
        agent.set_memory(MEMORY)
        agent.run(question = task)
    
    exit(0)
    
    
    # memory
    inpaths = [
        # "/robodata/taijing/RobotMem/data/captions/cobot/2025-03-10-17-01-55_VILA1.5-8b_3_secs.json",
        # "/robodata/taijing/RobotMem/data/captions/cobot/2025-03-10-17-00-15_VILA1.5-8b_3_secs.json",
        "/robodata/taijing/RobotMem/data/captions/cobot/2025-04-02-20-28-22_VILA1.5-8b_3_secs.json",
        "/robodata/taijing/RobotMem/data/captions/cobot/2025-04-02-20-30-36_VILA1.5-8b_3_secs.json",
    ]
    MEMORY = MilvusMemory("test2", obs_savepth=OBS_SAVEPATH, db_ip='127.0.0.1')
    MEMORY.reset()
    t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400
    remember_from_paths(MEMORY, inpaths, t_offset, viddir="/robodata/taijing/RobotMem/data/images")
    
    if args.replay:
        inpaths = [
            "/robodata/taijing/ros_perception/data/cobot/cobot_test_1/2025-04-01_13-53-08/2025-04-01_13-53-08.jsonl"
        ]
        update_from_paths(MEMORY, inpaths)
    
    # grounding dino
    # GD_BOX_THRESHOLD = args.gd_box_threshold
    # GD_TEXT_THRESHOLD = args.gd_text_threshold
    # GD_TOKEN_SPANS = args.gd_token_spans
    # OBJECT_DETECTOR = load_grounding_dino_model(args.gd_config_file, args.gd_checkpoint_path)
    
    # start agent
    rospy.init_node("remote_agent")
    threading.Thread(target=start_services, daemon=True).start()
    agent = Agent()
    agent.set_memory(MEMORY)
    
    rospy.sleep(0.5)
    rospy.loginfo("Finish loading...")
    
    tasks = [
        "Bring me a cup.",
        # "Bring me a white paper cup with a green Starbucks logo, featuring a plastic lid and handwritten text on the side.",
        "Today is Feb 08, 2025. Bring me the cup that was on the table yesterday.",
        "Today is Feb 08, 2025. Bring me the cup that was using with a plate yesterday.",
        "Today is Feb 08, 2025. Bring me the object that was on shelf yesterday.",
        "Bring me the cup that is usually on the table.",
    ]
    agent.run(question = tasks[-1])
    for task in tasks:
        agent.run(question=task)
    time.sleep(1)
    exit(0)
    
    if args.replay:
        agent.run(question="Bring me a cup")
    else:
        from math import radians
        # request_get_image_at_pose_service(11.5, 60, radians(135))
        # rospy.loginfo("finish navigating to waypoint1")
        # rospy.sleep(0.5)
        # request_get_image_at_pose_service(7.5, 60.9, radians(90))
        # rospy.loginfo("finish navigating to waypoint2")
        # rospy.sleep(0.5)
        # request_get_image_at_pose_service(11.7, 60, radians(45))
        # rospy.loginfo("finish navigating to waypoint3")
        # rospy.sleep(0.5)
        
        # COBOT.navigate(11.5, 59, radians(135))
        # COBOT.navigate(7.5, 60.9, radians(90)) 
        # COBOT.navigate(11.5, 60, radians(0))
        
        request_get_image_at_pose_service(11.5, 59, radians(135))
        rospy.loginfo("finish navigating to waypoint1")
        rospy.sleep(0.5)
        request_get_image_at_pose_service(7.5, 60.9, radians(90))
        rospy.loginfo("finish navigating to waypoint2")
        rospy.sleep(0.5)
        request_get_image_at_pose_service(11.7, 60.5, radians(0))
        rospy.loginfo("finish navigating to waypoint3")
        rospy.sleep(0.5)
        
        agent.run(question="Bring me a cup")
        rospy.sleep(20)
        rospy.spin()
        # 7.987890243530273, 61.08519744873047, -0.10746081173419952
        
    