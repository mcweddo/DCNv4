#!/bin/bash

set -e

# ===== Environment Variable Checks =====
if [[ -z "$GIT_REPO_URL" || -z "$S3_BUCKET" || -z "$S3_PREFIX" || -z "$S3_DATASET_CHECK_KEY" ]]; then
  echo "❌ Missing one or more required environment variables: GIT_REPO_URL, S3_BUCKET, S3_PREFIX, S3_DATASET_CHECK_KEY"
  exit 1
fi

# ===== Argument Validation =====
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <target_local_folder>"
  exit 1
fi

LOCAL_FOLDER="$1"
REPO_NAME=$(basename "$GIT_REPO_URL" .git)

# ===== Step 0: Install awscli and boto3 =====
echo "📦 Installing AWS CLI and boto3..."
pip install --upgrade pip
pip install awscli boto3

# ===== Step 1: Clone Git Repository =====
echo "📦 Cloning repository..."
if [[ ! -d "$REPO_NAME" ]]; then
  git clone "$GIT_REPO_URL"
else
  echo "✅ Repo already cloned: $REPO_NAME"
fi

# ===== Step 2: Check if Dataset Exists =====
echo "📥 Checking dataset presence..."
python3 <<EOF
import boto3, os, sys

s3 = boto3.client("s3")
bucket = os.environ["S3_BUCKET"]
check_key = os.environ["S3_DATASET_CHECK_KEY"]

try:
    s3.head_object(Bucket=bucket, Key=check_key)
    print("✅ Dataset already exists. Skipping download.")
    sys.exit(0)
except s3.exceptions.ClientError as e:
    if e.response["Error"]["Code"] == "404":
        print("📁 Dataset not found, starting download...")
    else:
        raise
EOF

# ===== Step 3: Download Dataset from S3 =====
python3 <<EOF
import boto3, os
from pathlib import Path

bucket = os.environ["S3_BUCKET"]
prefix = os.environ["S3_PREFIX"]
target_dir = Path("$LOCAL_FOLDER")

s3 = boto3.resource("s3")
bucket_obj = s3.Bucket(bucket)

for obj in bucket_obj.objects.filter(Prefix=prefix):
    if obj.key.endswith("/"):
        continue
    relative_path = Path(obj.key).relative_to(prefix)
    dest_path = target_dir / relative_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Downloading {obj.key} -> {dest_path}")
    bucket_obj.download_file(obj.key, str(dest_path))
EOF

# ===== Step 4: Install Python Requirements =====
echo "📦 Installing Python packages..."
pip install -U openmim
mim install mmcv-full==1.7.1
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.2
pip install opencv-python termcolor yacs pyyaml scipy
pip install 'numpy<2.0.0'
pip install pydantic==1.10.13
pip install yapf==0.40.1

# ===== Step 5: Build DCNv4 from Source =====
DCN_PATH="$REPO_NAME/DCNv4_op"

if [[ -d "$DCN_PATH" && -f "$DCN_PATH/make.sh" ]]; then
  echo "🧪 Checking build environment for DCNv4..."

  if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found. Please ensure CUDA toolkit is installed and nvcc is in PATH."
    exit 1
  fi

  python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'" || {
    echo "❌ CUDA is not available in PyTorch. Please check your PyTorch install."
    exit 1
  }

  GCC_VER=$(gcc -dumpversion | cut -d. -f1)
  if [[ "$GCC_VER" -gt 10 ]]; then
    echo "⚠️ Warning: GCC version $GCC_VER detected. GCC > 10 may cause issues compiling DCNv4."
  fi

  echo "🛠  Building DCNv4 from source at $DCN_PATH"
  cd "$DCN_PATH"
  bash make.sh
  python3 setup.py install
  cd -
else
  echo "❌ DCNv4 directory or make.sh not found at expected path: $DCN_PATH"
  exit 1
fi

echo "✅ All steps completed successfully!"
