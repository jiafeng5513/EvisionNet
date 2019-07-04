#!/usr/bin/env bash
/usr/local/anaconda3/bin/python3 ../Core/EvisionNet.py \
                --run_mode 2 \
                --test_seq 9 \
                --batch_size 1 \
                --seq_length 3 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                --output_dir ../test_output/test_pose/
# following https://github.com/tinghuiz/SfMLearner/issues/67,update by jiafeng5513