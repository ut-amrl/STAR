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

If you're using the MoMaBenchmark script, `<benchmark_dir>` referts to its `outputs` directory (the one contains annotations and tasks), and `<data_dir>` refers to the postprocessed `data` directory, containting captions and images.

To visualize the result:
```bash
python evaluation/analyze.py
```

In addition, you can also use `test_simulation.py` for a quick start. `--inpaths` assumes a list of paths to the caption json files, and `--viddirs` requires a list of paths to the corresponding image directory.