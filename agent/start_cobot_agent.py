import argparse
from PIL import Image as PILImage
import torch
import re
import os
import threading
import time

from agent import Agent
from utils.utils import ros_image_to_pil
from utils.memloader import remember_from_paths
from memory.memory import MilvusMemory, MemoryItem

import rospy
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import RememberSrv, RememberSrvResponse

CAPTIONER = None
MEMORY = None

def parse_args():
    default_query = "<video>\n You are a wandering around a household area. Please describe in detail what you see in the few seconds of the video. \
        Focus on foreground objects, events/activities, people and their actions, and other notable details. You can ignore anything that appeared to be further away from the camera (in the background). \
        Provide enough detail about objects (e.g., colors, patterns, logos, or states) to ensure they can be identified through text alone. For example,  Instead of just 'a box,' describe its color, any images or logos on it, and any distinguishing marks. \
        Think step by step about these details and be very specific. \
        Describe the video directly without any introductory phrases or extra commentary."
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    return args


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

def handle_observation_request(req):
    # rospy.loginfo(f"[Memory] Received observation request")
    
    timestamp = req.timestamp
    position = [req.x, req.y, req.theta]
    theta = req.theta
    
    pil_images = []
    for imgmsg in req.video:
        pil_image = ros_image_to_pil(imgmsg)
        pil_images.append(pil_image)
    caption = CAPTIONER.caption(pil_images, PROMPT)
    
    filenames = sorted(os.listdir(SAVEPATH))
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
        vidpath=SAVEPATH,
        start_frame=start_frame,
        end_frame=end_frame
    )
    MEMORY.insert(item, images=pil_images)
    
    return True

def milvus_observation_service():
    rospy.Service('/memory/observe', RememberSrv, handle_observation_request)
    rospy.loginfo("Memory observation service is ready.")
    rospy.spin()

if __name__ == "__main__":
    args = parse_args()
    PROMPT = args.query
    SAVEPATH = args.obs_savepath; os.makedirs(SAVEPATH, exist_ok=True)
    
    CAPTIONER = VILACaptioner(args)
    
    MEMORY = MilvusMemory("test1", obs_savepth=SAVEPATH, db_ip='127.0.0.1')
    MEMORY.reset()
    inpaths = [
        "/robodata/taijing/RobotMem/data/captions/cobot/2025-03-10-17-01-55_VILA1.5-8b_3_secs.json",
        "/robodata/taijing/RobotMem/data/captions/cobot/2025-03-10-17-00-15_VILA1.5-8b_3_secs.json",
    ]
    t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400
    remember_from_paths(MEMORY, inpaths, t_offset, viddir="/robodata/taijing/RobotMem/data/images")
    
    rospy.init_node("remote_agent")
    
    agent = Agent()
    agent.set_memory(MEMORY)
    
    threading.Thread(target=milvus_observation_service, daemon=True).start()
    
    while True:
        import time
        time.sleep(10)
        import pdb; pdb.set_trace()