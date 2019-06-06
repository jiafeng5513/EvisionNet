#!/usr/bin/env bash
/usr/local/anaconda3/bin/python3 test_kitti_pose.py \
                --test_seq 3 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                --output_dir ./test_output/test_pose/ \
                --ckpt_file ./checkpoints/model-191178