PWD=$(pwd)

cd $PWD/3dparty/VILA
git checkout ec7fb2c264920bf004fd9fa37f1ec36ea0942db5

pip install "setuptools<66.0.0"
pip install -e ".[train,eval]"
pip install triton==3.1.0
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
pip install protobuf==3.20.*

pip install langchain-community langgraph langchain_openai langchain_nvidia_ai_endpoints pymilvus gradio==3.50.2 langchain_huggingface sentence-transformers accelerate==0.33.0 deepspeed==0.9.5 pydantic==1.10.18

cd $PWD