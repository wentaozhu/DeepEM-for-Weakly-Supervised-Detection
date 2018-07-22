#!/bin/bash
set -e
cd weakdetectorv2
f=0 #7
i=47 #4
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model res18 -b 48 --resume results/res18/weakfd$f/0$i.ckpt --test 1 --testthresh -3 --save-dir res18/weakfd$f/ --config config_testtchi
if [ ! -d "results/res18/weakfd$f/traintchi/" ]; then
    mkdir results/res18/weakfd$f/traintchi/
fi
mv results/res18/weakfd$f/bbox/*.npy results/res18/weakfd$f/traintchi/