#!/usr/bin/env bash
set -x

GPUS=${GPUS:-8}
PORT=${PORT:-29500}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-12}

PY_ARGS=${@:2}  # Any other arguments 
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
    train.py \
    --with_box_refine \
    --dataset_file all \
    --binary \
    --batch_size 2 \
    --epochs 12 \
    --lr_drop 8 10 \
    ${PY_ARGS}
