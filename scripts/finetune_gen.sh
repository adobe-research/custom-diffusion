#!/usr/bin/env bash
#### command to run with retrieved images as regularization
# 1st arg: target caption
# 2nd arg: path to target images
# 3rd arg: path where generated images are saved
# 4rth arg: name of the experiment
# 5th arg: config name
# 6th arg: pretrained model path

ARRAY=()

for i in "$@"
do 
    echo $i
    ARRAY+=("${i}")
done


python -u sample.py \
        --n_samples 10 \
        --n_iter 20 \
        --scale 6 \
        --ddim_steps 50  \
        --ckpt ${ARRAY[5]} \
        --ddim_eta 1. \
        --outdir "${ARRAY[2]}" \
        --prompt "photo of a ${ARRAY[0]}" 

python -u  train.py \
        --base configs/custom-diffusion/${ARRAY[4]}  \
        -t --gpus 0,1 \
        --resume-from-checkpoint-custom ${ARRAY[5]} \
        --caption "<new1> ${ARRAY[0]}" \
        --datapath ${ARRAY[1]} \
        --reg_datapath "${ARRAY[2]}/samples" \
        --reg_caption "${ARRAY[0]}" \
        --modifier_token "<new1>" \
        --name "${ARRAY[3]}-sdv4"

