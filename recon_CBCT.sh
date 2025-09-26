#!/bin/bash

TASK="SVCT"
DEGREE=20

GPU=0
DATA="/your/data/path"

SLICE_BEGIN=0
SLICE_END=30
SINO_NOISE=0
RECON_SIZE=256

METHOD=DDS

NFE=30
NUM_CG=10
W_DPS=0
W_TIK=0
W_DZ=0
NOISE_CONTROL=None

USE_INIT=True
SIGMA_MAX=5

RENOISE_METHOD=DDPM

CHECKPOINT_PATH="/your/checkpoint/path/checkpoint_70.pth"
CONFIG_PATH="configs/ve/AAPM_256_ncsnpp_Chung.yaml"


python recon_CBCT.py \
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