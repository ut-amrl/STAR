PWD=$(pwd)

cd $PWD/3dparty/GroundingDINO

pip install -e .

mkdir -p weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

cd $PWD