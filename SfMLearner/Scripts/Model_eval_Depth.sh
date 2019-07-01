#!/usr/bin/env bash
/usr/local/anaconda3/bin/python3 ../Core/kitti_eval/eval_depth.py \
            --kitti_dir=/home/RAID1/DataSet/KITTI/KittiRaw/ \
            --pred_file=../Core/kitti_eval/kitti_eigen_depth_predictions.npy
