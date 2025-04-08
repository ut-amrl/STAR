#!/bin/bash

PWD=$(pwd)

cd 3dparty/GroundingDINO

python demo/detect_object_ros.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -v --box_threshold 0.5

cd $PWD