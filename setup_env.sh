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

# ===== Step 3: download with per-file speed bars  ==========================
python3 <<'PY'
import boto3, os, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from botocore.exceptions import BotoCoreError, ClientError

bucket      = os.environ["S3_BUCKET"]
prefix      = os.environ["S3_PREFIX"].rstrip("/") + "/"
repo_root   = Path("$REPO_NAME")
det_root    = repo_root / "detection"
marker_file = Path("$MARKER_FILE")

last_prefix_folder = Path(prefix.rstrip("/")).parts[-1]

s3    = boto3.resource("s3")
s3c   = boto3.client("s3")
bkt   = s3.Bucket(bucket)

objects = [
    (obj.key, det_root / last_prefix_folder / Path(obj.key).relative_to(prefix))
    for obj in bkt.objects.filter(Prefix=prefix)
    if not obj.key.endswith("/")
]

def download_with_retry(key, local_path, max_retries=3):
    if local_path.exists():
        return True

    size = s3c.head_object(Bucket=bucket, Key=key)["ContentLength"]
    bar  = tqdm(
        total=size, unit="B", unit_scale=True, unit_divisor=1024,
        desc=str(local_path.relative_to(det_root)), leave=False,
        position=threading.get_ident() % 64   # keep bars separate
    )

    def cb(bytes_amount):                 # tqdm callback
        bar.update(bytes_amount)

    for attempt in range(max_retries):
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                bkt.download_fileobj(key, f, Callback=cb)
            bar.close()
            return True
        except (BotoCoreError, ClientError):
            bar.close()
            time.sleep(2 ** attempt)
            if attempt == max_retries - 1:
                return key

failed = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(download_with_retry, k, p): k for k, p in objects}
    for _ in tqdm(as_completed(futs), total=len(futs),
                  desc="⬇️  files done", unit="file"):
        res = _.result()
        if res is not True:
            failed.append(res)

if failed:
    print("❌ failed files:", *failed, sep="\n  - ")
    raise RuntimeError("Download incomplete.")
else:
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    marker_file.write_text("Download completed.")
    print("✅ all files downloaded")
PY
# ========================================================================

COCO_ZIP_DIR="$REPO_NAME/detection/data/coco"

if [[ -d "$COCO_ZIP_DIR" ]]; then
  echo "📦 Scanning $COCO_ZIP_DIR for zip archives…"
  find "$COCO_ZIP_DIR" -maxdepth 1 -type f -name '*.zip' | while read -r zipfile; do
      echo "🗜️  Unzipping $(basename "$zipfile")"
      unzip -q -o "$zipfile" -d "$COCO_ZIP_DIR" && rm -f "$zipfile"
      #                                  └──────── remove zip only if unzip succeeded
  done
else
  echo "⚠️  Folder $COCO_ZIP_DIR not found – skipping unzip step."
fi

# ===== Step 4: Install Python Requirements =====
echo "📦 Installing Python packages..."
pip install -U openmim
mim install mmcv-full==1.5.0
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
