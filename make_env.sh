#!/bin/bash

set -e

export GIT_REPO_URL=https://github.com/mcweddo/DCNv4.git
export S3_BUCKET=
export S3_PREFIX=
export S3_DATASET_CHECK_KEY=path/to/dataset/.download_complete
export NCCL_P2P_DISABLE="1"

export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_DEFAULT_REGION=us-east-1