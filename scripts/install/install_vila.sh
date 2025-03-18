#!/bin/bash

PWD=$(pwd)

cd $PWD/3dparty/VILA

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install -e .
pip install -e ".[train]"
pip install -e ".[eval]"
pip install triton==3.1.0
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
pip install protobuf==3.20.*

pip install langchain-community langgraph langchain_openai langchain_nvidia_ai_endpoints pymilvus gradio==3.50.2 langchain_huggingface sentence-transformers accelerate==0.33.0 deepspeed==0.9.5 pydantic==1.10.18

cd $PWD