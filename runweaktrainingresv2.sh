#!/bin/bash
set -e

# python prepare.py
cd weakdetectorv2
maxeps=50
f=4 #2 ../weakdetector/results/res18/weakfd$f/weak029.ckpt
# Training
# CUDA_VISIBLE_DEVICES=0,2,3,1 python main.py --model res18 -b 32 --resume ../weakdetector/results/res18/weakfd$f/weak024.ckpt --start-epoch 1 --save-dir res18/weakfd$f/ --epochs $maxeps --config config_trainingweak$f
# Inference
for (( i=1; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    if [ $i -lt 10 ]; then
        CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model res18 -b 32 --resume results/res18/weakfd$f/00$i.ckpt --test 1 --save-dir res18/weakfd$f/ --config config_trainingweak$f
    elif [ $i -lt 100 ]; then 
        CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model res18 -b 32 --resume results/res18/weakfd$f/0$i.ckpt --test 1 --save-dir res18/weakfd$f/ --config config_trainingweak$f
    elif [ $i -lt 1000 ]; then
        CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model res18 -b 32 --resume results/res18/weakfd$f/$i.ckpt --test 1 --save-dir res18/weakfd$f/ --config config_trainingweak$f
    else
        echo "Unhandled case"
    fi
    if [ ! -d "results/res18/weakfd$f/val$i/" ]; then
        mkdir results/res18/weakfd$f/val$i/
    fi
    mv results/res18/weakfd$f/bbox/*.npy results/res18/weakfd$f/val$i/
done 