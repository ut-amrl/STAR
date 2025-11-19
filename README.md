# STAR - Searching in Space and Time: Unified Memory-Action Loops for Open-World Object Retrieval

<div align="center">

[![Website](https://img.shields.io/badge/Website-STAR-blue.svg)](https://tiejean.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2511.14004-b31b1b.svg)](https://arxiv.org/abs/2511.14004)


**Taijing Chen**, **Sateesh Kumar**, **Junhong Xu**, **Georgios Pavlakos**, **Joydeep Biswas**, **Roberto Martin-Martin**

<img src="static/overview.gif" alt="STAR teaser animation" width="100%">

</div>

---

## Quick Start
```bash
bash scripts/install/install_milvus.sh
bash standalone_embed.sh start
docker pull docker.io/tiejean/star-ros-noetic:dev
podman run --rm -it \
    --name star-ros-noetic \
    --network host \
    --env-file .env \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v <host-start-project-root>:<client-start-project-root>:rw,z -e USE_PY310=1  \
    star-ros-noetic:dev bash
python quick_start_server.py # Replace the dummy service with the correct one
```
In another terminal:
```bash
docker exec -it --workdir <client-start-project-root>  -e USE_PY310=1 moma-ros-noetic bash -l
```

## Project Directory
TODO; all instructions, unless specified otherwise, are assumed to be run under project root directory.

## Setup
We provide docker file tested on cuda 12.6 and ROS1 noetic.


### Milvus (Vector) Database Docker Setup
STAR stores memory in Milvus Vector Database. To install Milvus:
```bash
bash scripts/install/install_milvus.sh
```

To start Milvus Database (required before starting STAR):
```bash
bash standalone_embed.sh start
```
And you should see "Milvus is running." on terminal.

#### Known Issues and Fixes
If the container is already running but the connection is broken, restart the vector database server:
```bash
bash standalone_embed.sh restart
```

If you encounter error "Failed to start transient timer unit: Unit <container_id>.service", run:
```
docker kill <container_id> && docker rm <container_id>
```
And then run:
```bash
bash standalone_embed.sh start
```


### STAR Docker Setup
To build the image:
```bash
cd docker
docker build -t star-ros-noetic:dev -f Dockerfile --build-arg CUDA_VERSION=12.6.0 --pull .
```

Next, create an `.env` file under `docer` and add `OPENAI_API_KEY` to `.env`. STAR has been tested on ChatGPT o3.

To start the container:
```bash
podman run --rm -it \
    --hooks-dir=/usr/share/containers/oci/hooks.d \
    --network host \
    --env-file .env \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v <host-start-project-root>:<client-start-project-root>:rw,z \
    -e USE_PY310=1  \
    star-ros-noetic:dev bash
```
Mount any other directories you need to use through `-v` flag. For example:
```bash
podman run --rm -it \
    --hooks-dir=/usr/share/containers/oci/hooks.d \
    --network host \
    --env-file .env \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v <host-STAR-project-root>:<client-STAR-project-root>:rw,z \
    -v <host-datapath>:<client-datapath>:rw,z \
    -v <host-task-metadata>:<client-task-metadata>:rw,z \
    -e USE_PY310=1  \
    star-ros-noetic:dev bash
```

## Running STAR
Check out the toy example. (Instructions comming soon).

## STARBench
Comming soon.

## Citation
```
@misc{chen2025searchingspacetimeunified,
      title={Searching in Space and Time: Unified Memory-Action Loops for Open-World Object Retrieval}, 
      author={Taijing Chen and Sateesh Kumar and Junhong Xu and George Pavlakos and J oydeep Biswas and Roberto Martín-Martín},
      year={2025},
      eprint={2511.14004},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.14004}, 
}
```