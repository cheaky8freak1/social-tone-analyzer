#!/bin/bash

ONNX_MODEL="models/multimodal_classifier.onnx"
PLAN_MODEL="models/multimodal_classifier.plan"

docker run --gpus all -it --rm \
  -v $(pwd)/models:/workspace/models \
  nvcr.io/nvidia/tensorrt:24.10-py3 \
  trtexec \
    --onnx=/workspace/$ONNX_MODEL \
    --saveEngine=/workspace/$PLAN_MODEL \
    --fp16 \
    --minShapes=img_emb:1x512,txt_emb:1x768 \
    --optShapes=img_emb:8x512,txt_emb:8x768 \
    --maxShapes=img_emb:32x512,txt_emb:32x768 \
    --workspace=4096
