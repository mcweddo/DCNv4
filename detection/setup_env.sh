#!/bin/bash

set -e

# ===== Environment Variable Checks =====
if [[ -z "$GIT_REPO_URL" || -z "$S3_BUCKET" || -z "$S3_PREFIX" || -z "$S3_DATASET_CHECK_KEY" ]]; then
  echo "❌ Missing one or more required environment variables: GIT_REPO_URL, S3_BUCKET, S3_PREFIX, S3_DATASET_CHECK_KEY"
  exit 1
fi

REPO_NAME=$(basename "$GIT_REPO_URL" .git)
TARGET_FOLDER="$REPO_NAME/detection"
MARKER_FILE="$TARGET_FOLDER/.download_complete"

# ===== Step 0: Install awscli and boto3 =====
echo "📦 Installing AWS CLI and boto3..."
pip install --upgrade pip
pip install awscli boto3 tqdm

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

# ===== Step 3: Download Dataset from S3 with retries & final path logic =====
python3 <<EOF
import boto3, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib.parse import urlparse
from botocore.exceptions import BotoCoreError, ClientError

bucket = os.environ["S3_BUCKET"]
prefix = os.environ["S3_PREFIX"].rstrip("/") + "/"
repo_root = Path("$REPO_NAME")
detection_root = repo_root / "detection"
marker_file = Path("$MARKER_FILE")

# Extract last folder in prefix
last_prefix_folder = Path(prefix.rstrip("/")).parts[-1]

s3 = boto3.resource("s3")
bucket_obj = s3.Bucket(bucket)

# Create key-to-local path mapping
objects = [
    (
        obj.key,
        detection_root / last_prefix_folder / Path(obj.key).relative_to(prefix)
    )
    for obj in bucket_obj.objects.filter(Prefix=prefix)
    if not obj.key.endswith("/")
]

def download_with_retry(key, local_path, max_retries=3):
    if local_path.exists():
        return True  # Skip if already exists
    local_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            print("ima")
            bucket_obj.download_file(key, str(local_path))
            return True
        except (BotoCoreError, ClientError):
            time.sleep(2 ** attempt)
            if attempt == max_retries - 1:
                return key  # Return key on final failure

# Run downloads in parallel
failed = []
with ThreadPoolExecutor(max_workers=8) as executor:
    future_to_key = {
        executor.submit(download_with_retry, key, path): key
        for key, path in objects
    }
    for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="⬇️  Downloading from S3"):
        result = future.result()
        if result is not True:
            failed.append(result)

if failed:
    print("❌ The following files failed to download after retries:")
    for f in failed:
        print(f" - {f}")
    raise RuntimeError("Download incomplete. Please retry.")
else:
    marker_file.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists
    marker_file.write_text("Download completed.")
    print("✅ All files downloaded successfully.")
EOF




# ===== Step 4: Install Python Requirements =====
echo "📦 Installing Python packages..."
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
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
