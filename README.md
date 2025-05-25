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
To run evaluation script:
```bash
python evaluate.py --benchmark_dir <benchmark_dir> --data_dir <data_dir> --task_types <unambiguous spatial spatial_temporal ...>
```