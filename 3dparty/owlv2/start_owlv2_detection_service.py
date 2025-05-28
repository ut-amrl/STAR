import os
import argparse
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

import rospy
from sensor_msgs.msg import Image
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.msg import BBox2DMsg, BBox2DArrayMsg
from amrl_msgs.srv import SemanticObjectDetectionSrv, SemanticObjectDetectionSrvResponse

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16").to(device)
threshold = 0.3  # Default threshold, can be overridden by command line argument

def parse_args():
    parser = argparse.ArgumentParser(description="Start OWLv2 Object Detection Service")
    # Object Detection parameters
    parser.add_argument("--threshold", type=float, default=0.3, help="box threshold")
    args = parser.parse_args()
    return args

def visualize_detections_pil(image, result, save_path="debug/pred.png"):
    """
    Draw bounding boxes and labels on the image using PIL and save it.

    Args:
        image (PIL.Image): Original input image.
        result (dict): Output from OWLv2 post_process_grounded_object_detection
        save_path (str): File path to save the visualization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["text_labels"]

    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box.tolist()
        label_text = f"{label} ({score:.2f})"

        # Draw box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        # Draw label background and text
        text_size = font.getbbox(label_text)[2:]
        draw.rectangle([xmin, ymin - text_size[1], xmin + text_size[0], ymin], fill="red")
        draw.text((xmin, ymin - text_size[1]), label_text, fill="white", font=font)

    image.save(save_path)

def ros_image_to_pil(ros_image):
    np_arr = np.frombuffer(ros_image.data, dtype=np.uint8)
    if ros_image.encoding == "rgb8" or ros_image.encoding == "bgr8":
        image = np_arr.reshape((ros_image.height, ros_image.width, 3))
        if ros_image.encoding == "bgr8":
            image = image[:, :, ::-1]  # Convert BGR → RGB
    else:
        raise ValueError(f"Unsupported encoding: {ros_image.encoding}")
    return PILImage.fromarray(image)

def handle_object_detection_request(req):
    rospy.loginfo(f"[OWLv2] Received request: '{req.query_text}'")

    
    query_pil_img = ros_image_to_pil(req.query_image)
    text_labels = [[req.query_text]]  # OWLv2 expects nested list

    # Preprocess
    inputs = processor(text=text_labels, images=query_pil_img, return_tensors="pt").to(device)

    # Forward
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocess
    target_sizes = torch.tensor([(query_pil_img.height, query_pil_img.width)]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold,
        text_labels=text_labels
    )
    result = results[0]

    # Format ROS response
    bbox_arr_msg = BBox2DArrayMsg(header=req.query_image.header)

    for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        bbox = BBox2DMsg(label=label, conf=score.item(), xyxy=[xmin, ymin, xmax, ymax])
        bbox_arr_msg.bboxes.append(bbox)

    rospy.loginfo(f"[OWLv2] Returning {len(result['boxes'])} detections.")
    # Save debug visualization
    visualize_detections_pil(query_pil_img.copy(), result, save_path="debug/pred.png")
    
    return SemanticObjectDetectionSrvResponse(bounding_boxes=bbox_arr_msg)

def owl_service():
    rospy.init_node('owlv2_object_detector')
    rospy.Service('/owlv2/semantic_object_detection', SemanticObjectDetectionSrv, handle_object_detection_request)
    rospy.loginfo("OWLv2 detection service is ready.")
    rospy.spin()

if __name__ == "__main__":
    args = parse_args()
    threshold = args.threshold # Global threshold set by CLI
    owl_service()