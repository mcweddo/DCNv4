#!/usr/bin/env bash

set -euo pipefail

# Detect Ubuntu release (e.g. focal, jammy)
DISTRO=$(lsb_release -cs)

echo "Adding NVIDIA TensorRT apt repository for Ubuntu ${DISTRO}…"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release

curl -sSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${DISTRO}/x86_64/7fa2af80.pub \
  | sudo apt-key add -

curl -sSL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu${DISTRO}/x86_64/7fa2af80.pub \
  | sudo apt-key add -

echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu${DISTRO}/x86_64/ /" \
  | sudo tee /etc/apt/sources.list.d/tensorrt.list

sudo apt-get update

echo "Installing TensorRT development packages…"
sudo apt-get install -y --no-install-recommends \
    libnvinfer8 \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libnvonnxparsers-dev \
    libnvparsers-dev \
    libnvrtc-dev

echo "Installing ONNX & Protobuf development packages…"
sudo apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    libprotoc-dev \
    libonnx-dev

echo "Installing Python ONNX tooling (onnx, onnxruntime, onnx-simplifier)…"
python3 -m pip install --upgrade --no-cache-dir \
    onnx \
    onnxruntime \
    onnx-simplifier

echo "All dependencies installed successfully."
