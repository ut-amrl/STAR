import requests
import os, time
from PIL import Image, ImageDraw, ImageFont
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection

def visualize_detections_pil(image, result, save_path="debug/pred.png"):
    """
    Draw bounding boxes and labels on the image using PIL and save it.

    Args:
        image (PIL.Image): Original input image.
        result (dict): Output dictionary from `processor.post_process_grounded_object_detection`.
        save_path (str): Where to save the visualization.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    draw = ImageDraw.Draw(image)
    
    # Optional: load default font
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
        # Draw label background
        text_size = font.getbbox(label_text)[2:]
        draw.rectangle([xmin, ymin - text_size[1], xmin + text_size[0], ymin], fill="red")
        # Draw label text
        draw.text((xmin, ymin - text_size[1]), label_text, fill="white", font=font)

    image.save(save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text_labels = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=text_labels, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

start_time = time.time()
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.tensor([(image.height, image.width)])
# Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
results = processor.post_process_grounded_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
)
end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")

# Retrieve predictions for the first image for the corresponding text queries
result = results[0]
boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
for box, score, text_label in zip(boxes, scores, text_labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
    
visualize_detections_pil(image, result, save_path="debug/pred.png")