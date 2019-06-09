#!usr/bin/env bash
# select gpu devices
export CUDA_VISIBLE_DEVICES=0,1
# train
# ../data/voc/ is the path of VOCdevkit.
python -m experiment.demo_voc2007 --batch-size 16 --lr 1e-3 --epochs 20 
