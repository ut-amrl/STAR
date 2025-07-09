import os
import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO

# --- Configuration ---
image_paths = [
    "debug/toy_examples/000005.png",
    "debug/toy_examples/000035.png",
    "debug/toy_examples/000045.png"
]

client = OpenAI()  # Requires OPENAI_API_KEY in your env

# --- Helper: Convert local image to base64 data URI ---
def image_to_data_uri(image_path):
    with Image.open(image_path).convert("RGB") as img:
        img = img.resize((384, 384), Image.BILINEAR)  # ⬅️ Downscale here
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

# --- Single-image calls ---
print("🔍 Running single-image captioning:")
for i, path in enumerate(image_paths):
    image_data_uri = image_to_data_uri(path)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Please caption the image {os.path.basename(path)}"},
            {"type": "image_url", "image_url": {"url": image_data_uri}}
        ]}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    print(f"[{os.path.basename(path)}] Caption: {response.choices[0].message.content.strip()}")

# --- Multi-image call ---
print("\n🧪 Running multi-image captioning (all in one call):")
multi_messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "Please caption each of the following 3 images. Label them as Image 1, 2, and 3 in your response."},
        *[
            {"type": "image_url", "image_url": {"url": image_to_data_uri(p)}}
            for p in image_paths
        ]
    ]}
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=multi_messages
)
print("[Combined] Response:\n", response.choices[0].message.content.strip())

print("\n🧪 Running caption test on dumped memory_messages:")
from pathlib import Path
import json

dump_dir = Path("debug/dumped_memory_messages")
with open(dump_dir / "messages.json", "r") as f:
    msg_defs = json.load(f)

message = {"role": "user", "content": []}

for item in msg_defs:
    if item["type"] == "text":
        with open(dump_dir / item["file"], "r") as f:
            message["content"].append({
                "type": "text",
                "text": f.read()
            })
    elif item["type"] == "image":
        with open(dump_dir / item["file"], "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            message["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"}
            })

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[message]
)
print("[From dumped memory messages] Response:\n", response.choices[0].message.content.strip())

