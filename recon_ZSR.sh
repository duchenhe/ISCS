#!/bin/bash

TASK="ZSR"
DEGREE=5

GPU=0
DATA="/your/data/path"

SLICE_BEGIN=0
SLICE_END=0
SINO_NOISE=0
RECON_SIZE=256

METHOD=DDS
NFE=5
NUM_CG=1
W_DPS=0
W_TIK=0
W_DZ=0
NOISE_CONTROL=SLERP

USE_INIT=True
SIGMA_MAX=5

RENOISE_METHOD=DDPM

CHECKPOINT_PATH="/your/checkpoint/path/checkpoint_70.pth"
CONFIG_PATH="configs/ve/BMR_ZSR_256.yaml"

python recon_MRI_ZSR.py \
--method $METHOD \
--task $TASK \
--degree $DEGREE \
--gpu $GPU \
--data $DATA \
--slice-begin $SLICE_BEGIN \
--slice-end $SLICE_END \
--sino-noise $SINO_NOISE \
--recon-size $RECON_SIZE \
--NFE $NFE \
--num-cg $NUM_CG \
--w-dps $W_DPS \
--w-tik $W_TIK \
--w-dz $W_DZ \
--noise-control $NOISE_CONTROL \
--use-init $USE_INIT \
--sigma-max $SIGMA_MAX \
--renoise-method $RENOISE_METHOD \
--checkpoint-path $CHECKPOINT_PATH \
--config-path $CONFIG_PATH \