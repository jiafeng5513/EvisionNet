#!/usr/bin/env bash
/usr/local/anaconda3/bin/python3 test_kitti_depth.py \
            --dataset_dir /home/RAID1/DataSet/KITTI/KittiRaw/ \
            --output_dir ./test_output/test_depth \
            --ckpt_file ./checkpoints/model-191178