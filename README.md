To install dependency, under project root directory, run:
```bash
bash scripts/install_all.sh
```
You're expected to see broken dependency error messages. Most are fine, but make sure "python -c "import torch; print(torch.version.cuda)"` and `nvcc --version` report the same pytorch version.

## Vector Database
First, go in to `vector_db` directory:
```bash
cd vector_db
```

To verify vector database for first time:
```bash
bash standalone_embed.sh start
```

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
