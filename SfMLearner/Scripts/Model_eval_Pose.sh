#!/usr/bin/env bash
/usr/local/anaconda3/bin/python3 ../Core/kitti_eval/eval_pose.py \
            --gtruth_dir=../Core/kitti_eval/pose_data/ground_truth/09/ \
            --pred_dir=../Core/kitti_eval/pose_data/ours_results/09/

/usr/local/anaconda3/bin/python3 ../Core/kitti_eval/eval_pose.py \
            --gtruth_dir=../Core/kitti_eval/pose_data/ground_truth/10/ \
            --pred_dir=../Core/kitti_eval/pose_data/ours_results/10/