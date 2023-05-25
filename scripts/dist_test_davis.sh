#!/usr/bin/env bash
set -x

GPUS=1

python3 inference_davis.py --with_box_refine --binary --freeze_text_encoder --ngpu ${GPUS} --backbone $1

