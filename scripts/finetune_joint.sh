#!/usr/bin/env bash
#### command to run with retrieved images as regularization
# 1st arg: target caption1
# 2nd arg: path to target images1
# 3rd arg: path where retrieved images1 are saved
# 4rth arg: target caption2
# 5th arg: path to target images2
# 6th arg: path where retrieved images2 are saved
# 7th arg: name of the experiment
# 8th arg: config name
# 9th arg: pretrained model path

ARRAY=()

for i in "$@"
do 
    echo $i
    ARRAY+=("${i}")
done


python src/retrieve.py --target_name "${ARRAY[0]}" --outpath ${ARRAY[2]}
python src/retrieve.py --target_name "${ARRAY[3]}" --outpath ${ARRAY[5]}


python -u  train.py \
        --base configs/custom-diffusion/${ARRAY[7]}  \
        -t --gpus 0,1 \
        --resume-from-checkpoint-custom  ${ARRAY[8]} \
        --caption "<new1> ${ARRAY[0]}" \
        --datapath ${ARRAY[1]} \
        --reg_datapath "${ARRAY[2]}/images.txt" \
        --reg_caption "${ARRAY[2]}/caption.txt" \
        --caption2 "<new2> ${ARRAY[3]}" \
        --datapath2 ${ARRAY[4]} \
        --reg_datapath2 "${ARRAY[5]}/images.txt" \
        --reg_caption2 "${ARRAY[5]}/caption.txt" \
        --modifier_token "<new1>+<new2>" \
        --name "${ARRAY[6]}-sdv4"

