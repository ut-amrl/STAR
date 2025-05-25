To install dependency, under project root directory, run:
```bash
bash scripts/install_all.sh
```
You're expected to see broken dependency error messages. Most are fine, but make sure "python -c "import torch; print(torch.version.cuda)"` and `nvcc --version` report the same pytorch version.

To verify vector database for first time:
```bash
cd vector_db && bash standalone_embed.sh start
```

If the container is already running but the connection is broken, restart the vector database server:
```bash
cd vector_db && bash standalone_embed.sh restart
```