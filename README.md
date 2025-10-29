# STAR: Searching in Space and Time

## Quick Start

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
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v <host-start-project-root>:<client-start-project-root>:rw,z \
    -v /robodata/taijing/benchmarks/virtualhome:/robodata/taijing/benchmarks/virtualhome:rw,z \
    -v /robodata/taijing/benchmarks/MoMaBenchmark:/robodata/taijing/benchmarks/MoMaBenchmark:rw,z \
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


## Evaluation
You will need to first start virtualhome ros service.
```
git clone https://github.com/TieJean/virtualhome.git
cd virtualhome
git checkout develop
```
Follow the instruction to install virtualhome.

Install arml_msgs:
```
git clone https://github.com/ut-amrl/amrl_msgs.git
cd amrl_msgs
git checkout taijing/moma/perception
```
Follow the instruction to install the messages and services.

To start virtualhome ros service -
1. Start ros1 in one terminal
```
roscore
```
2. Follow the instruction to start the virtualhome unity simulation in another terminal.
3. In the virtualhome repo:
```
cd virtualhome/demo
python start_ros_service.py
```

To evaluate:
```bash
python evaluate_sim.py --benchmark_dir <benchmark_dir> --data_dir <data_dir> --task_types <unambiguous spatial spatial_temporal ...> --agent_type <low_level_gt high_level_gt> <--force_rerun>
```
This script will evaluate all tasks under `config/tasks_sim.txt`. If `force_rerun` is not set, it will skip the already evaluated one.

If you're using the MoMaBenchmark script, `<benchmark_dir>` referts to its `outputs` directory (the one contains annotations and tasks), and `<data_dir>` refers to the postprocessed `data` directory, containting captions and images.

To visualize the result:
```bash
python evaluation/analyze.py
```
