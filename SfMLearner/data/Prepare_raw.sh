#!/usr/bin/env bash
/usr/local/anaconda3/bin/python3 data/prepare_train_data.py \
                  --dataset_dir=/home/RAID1/DataSet/KITTI/KittiRaw/ \
                  --dataset_name='kitti_raw_eigen' \
                  --dump_root=/home/RAID1/DataSet/KITTI/KittiRaw_prepared/ \
                  --seq_length=3 \
                  --img_width=416 \
                  --img_height=128 \
                  --num_threads=4