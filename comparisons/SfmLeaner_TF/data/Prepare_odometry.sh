#!/usr/bin/env bash
/usr/local/anaconda3/bin/python3 ./prepare_train_data.py \
        --dataset_dir=/home/RAID1/DataSet/KITTI/KittiOdometry/ \
        --dataset_name='kitti_odom' \
        --dump_root=/home/RAID1/DataSet/KITTI/KittiOdometry_prepared/ \
        --seq_length=3 \
        --img_width=416 \
        --img_height=128 \
        --num_threads=4
