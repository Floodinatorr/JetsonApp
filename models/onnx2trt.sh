#!/bin/bash

python -m venv model_env
source model_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
yolo export model=yolo11s.pt format=onnx device=0 half=True
deactivate