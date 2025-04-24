#!/usr/bin/env bash
set -euo pipefail

########################################################################
# 0 ── ENV-VAR CHECKS ──────────────────────────────────────────────────
########################################################################
for v in GIT_REPO_URL S3_BUCKET S3_PREFIX; do
  if [[ -z "${!v:-}" ]]; then
    echo "❌  Environment variable $v is not set." && exit 1
  fi
done

########################################################################
# 1 ── RESOLVE KEY PATHS ───────────────────────────────────────────────
########################################################################
# strip trailing ".git", any path parts, and possible '?' fragment
REPO_DIR=$(basename -s .git "${GIT_REPO_URL%%\?*}")
# last element of S3_PREFIX (e.g. "data" from ".../detection/data/")
LAST_PREFIX_FOLDER=$(basename "${S3_PREFIX%/}")

TARGET_BASE="${REPO_DIR}/detection/${LAST_PREFIX_FOLDER}"
MARKER_FILE="${TARGET_BASE}/.download_complete"

# make them visible inside Python
export REPO_DIR TARGET_BASE MARKER_FILE

########################################################################
# 2 ── TOOLING: pip, awscli, boto3, tqdm ───────────────────────────────
########################################################################
pip install --upgrade pip
pip install awscli boto3 tqdm

########################################################################
# 3 ── CLONE OR UPDATE THE GIT REPO ────────────────────────────────────
########################################################################
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "✅  Repo exists – pulling latest"
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$GIT_REPO_URL" "$REPO_DIR"
fi

########################################################################
# 4 ── DOWNLOAD DATASET (parallel, progress, retry, skip) ─────────────
########################################################################
if [[ -f "$MARKER_FILE" ]]; then
  echo "✅  Dataset already downloaded (marker found)."
else
python3 <<'PY'
import os, time, boto3, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from botocore.exceptions import BotoCoreError, ClientError

bucket        = os.environ['S3_BUCKET']
prefix        = os.environ['S3_PREFIX'].rstrip('/') + '/'
target_base   = Path(os.environ['TARGET_BASE'])
marker_file   = Path(os.environ['MARKER_FILE'])

s3  = boto3.resource('s3')
cli = boto3.client('s3')
bkt = s3.Bucket(bucket)

# build list of (key, local_path)
objects = [(obj.key,
            target_base / Path(obj.key).relative_to(prefix))
           for obj in bkt.objects.filter(Prefix=prefix)
           if not obj.key.endswith('/')]

def download(key, dst, retries=3):
    if dst.exists():
        return True
    total = cli.head_object(Bucket=bucket, Key=key)['ContentLength']
    bar   = tqdm(total=total,
                 desc=str(dst.relative_to(target_base)),
                 unit='B', unit_scale=True, leave=False,
                 position=threading.get_ident() % 64)

    def cb(chunk): bar.update(chunk)

    for i in range(retries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, 'wb') as fh:
                bkt.download_fileobj(key, fh, Callback=cb)
            bar.close(); return True
        except (BotoCoreError, ClientError):
            bar.close(); time.sleep(2**i)
    return key  # failed

failed=[]
with ThreadPoolExecutor(max_workers=8) as exe:
    fut = {exe.submit(download,k,p):k for k,p in objects}
    for f in tqdm(as_completed(fut), total=len(fut),
                  desc='⬇️  files done', unit='file'):
        if f.result() is not True:
            failed.append(f.result())

if failed:
    print('❌  Failed files:\n - ' + '\n - '.join(failed))
    raise SystemExit(1)

marker_file.parent.mkdir(parents=True, exist_ok=True)
marker_file.write_text('Download completed.')
print('✅  Dataset downloaded.')
PY
fi

########################################################################
# 5 ── UNZIP any *.zip in detection/data/coco/  then delete zips ——────
########################################################################
COCO_ZIP_DIR="$TARGET_BASE/coco"
if [[ -d "$COCO_ZIP_DIR" ]]; then
  find "$COCO_ZIP_DIR" -maxdepth 1 -type f -name '*.zip' | while read -r z; do
    echo "🗜️  Unzipping $(basename "$z")"
    unzip -q -o "$z" -d "$COCO_ZIP_DIR" && rm -f "$z"
  done
fi

########################################################################
# 6 ── PYTHON DEPENDENCIES ─────────────────────────────────────────────
########################################################################
pip install -U openmim
mim install mmcv-full==1.7.1        # works with torch ≥1.13
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.2 \
            opencv-python termcolor yacs pyyaml scipy \
            'numpy<2.0' pydantic==1.10.13 yapf==0.40.1

########################################################################
# 7 ── BUILD DCNv4 with GCC-9 (if present) ─────────────────────────────
########################################################################
if command -v gcc-9 &>/dev/null; then
  export CC=gcc-9 CXX=g++-9
fi
DCN_OP="${REPO_DIR}/DCNv4_op"
if [[ -d "$DCN_OP" && -f "$DCN_OP/make.sh" ]]; then
  echo "🛠  Building DCNv4 CUDA op …"
  pushd "$DCN_OP" >/dev/null
  bash make.sh
  python setup.py install
  popd >/dev/null
fi

echo -e "\n🎉  All steps finished successfully."
