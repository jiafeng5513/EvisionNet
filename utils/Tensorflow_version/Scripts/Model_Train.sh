#!/usr/bin/env bash
/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode=0 \
            --dataset_dir=/home/RAID1/DataSet/KITTI/KittiRaw_prepared/ \
            --checkpoint_dir=../checkpoints/ \
            --img_width=416 \
            --img_height=128 \
            --batch_size=16 \
            --num_gpus=4 \
            --learning_rate=0.0002 \
            --learning_rate_decay_factor=0.5 \
            --num_epochs_per_decay=5 \
            --beta1=0.9 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.0 \
            --batch_size=4 \
            --seq_length=3 \
            --num_source=2 \
            --num_epochs=30 \
            --log_prefix=1

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode 1 \
            --batch_size=16 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.0 \
            --seq_length=3 \
            --num_source=2 \
            --dataset_dir /home/RAID1/DataSet/KITTI/KittiRaw/ \
            --output_dir ../test_output/test_depth \
            --log_prefix=1

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 9 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.0 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                 --log_prefix=1

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 10 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.0 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                --log_prefix=1

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode=0 \
            --dataset_dir=/home/RAID1/DataSet/KITTI/KittiRaw_prepared/ \
            --checkpoint_dir=../checkpoints/ \
            --img_width=416 \
            --img_height=128 \
            --batch_size=4 \
            --num_gpus=4 \
            --learning_rate=0.0002 \
            --learning_rate_decay_factor=0.5 \
            --num_epochs_per_decay=5 \
            --beta1=0.9 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.0 \
            --batch_size=4 \
            --seq_length=3 \
            --num_source=2 \
            --num_epochs=30 \
            --log_prefix=2

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode 1 \
            --batch_size=4 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.0 \
            --seq_length=3 \
            --num_source=2 \
            --dataset_dir /home/RAID1/DataSet/KITTI/KittiRaw/ \
            --output_dir ../test_output/test_depth \
            --log_prefix=2

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 9 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.0 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                 --log_prefix=2

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 10 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.0 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                --log_prefix=2

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode=0 \
            --dataset_dir=/home/RAID1/DataSet/KITTI/KittiRaw_prepared/ \
            --checkpoint_dir=../checkpoints/ \
            --img_width=416 \
            --img_height=128 \
            --batch_size=4 \
            --num_gpus=4 \
            --learning_rate=0.0002 \
            --learning_rate_decay_factor=0.5 \
            --num_epochs_per_decay=5 \
            --beta1=0.9 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.1 \
            --batch_size=4 \
            --seq_length=3 \
            --num_source=2 \
            --num_epochs=30 \
            --log_prefix=3

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode 1 \
            --batch_size=4 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.1 \
            --seq_length=3 \
            --num_source=2 \
            --dataset_dir /home/RAID1/DataSet/KITTI/KittiRaw/ \
            --output_dir ../test_output/test_depth \
            --log_prefix=3

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 9 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.1 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                 --log_prefix=3

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 10 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.1 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                --log_prefix=3

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode=0 \
            --dataset_dir=/home/RAID1/DataSet/KITTI/KittiRaw_prepared/ \
            --checkpoint_dir=../checkpoints/ \
            --img_width=416 \
            --img_height=128 \
            --batch_size=4 \
            --num_gpus=4 \
            --learning_rate=0.0002 \
            --learning_rate_decay_factor=0.5 \
            --num_epochs_per_decay=5 \
            --beta1=0.9 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.05 \
            --batch_size=4 \
            --seq_length=3 \
            --num_source=2 \
            --num_epochs=30 \
            --log_prefix=4

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode 1 \
            --batch_size=4 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.05 \
            --seq_length=3 \
            --num_source=2 \
            --dataset_dir /home/RAID1/DataSet/KITTI/KittiRaw/ \
            --output_dir ../test_output/test_depth \
            --log_prefix=4

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 9 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.05 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                 --log_prefix=4

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 10 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.05 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                --log_prefix=4

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode=0 \
            --dataset_dir=/home/RAID1/DataSet/KITTI/KittiRaw_prepared/ \
            --checkpoint_dir=../checkpoints/ \
            --img_width=416 \
            --img_height=128 \
            --batch_size=2 \
            --num_gpus=4 \
            --learning_rate=0.0002 \
            --learning_rate_decay_factor=0.5 \
            --num_epochs_per_decay=5 \
            --beta1=0.9 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.05 \
            --batch_size=4 \
            --seq_length=3 \
            --num_source=2 \
            --num_epochs=30 \
            --log_prefix=5

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode 1 \
            --batch_size=2 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.05 \
            --seq_length=3 \
            --num_source=2 \
            --dataset_dir /home/RAID1/DataSet/KITTI/KittiRaw/ \
            --output_dir ../test_output/test_depth \
            --log_prefix=5

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 9 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.05 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                 --log_prefix=5

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 10 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.05 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                --log_prefix=5

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode=0 \
            --dataset_dir=/home/RAID1/DataSet/KITTI/KittiRaw_prepared/ \
            --checkpoint_dir=../checkpoints/ \
            --img_width=416 \
            --img_height=128 \
            --batch_size=1 \
            --num_gpus=4 \
            --learning_rate=0.0002 \
            --learning_rate_decay_factor=0.5 \
            --num_epochs_per_decay=5 \
            --beta1=0.9 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.05 \
            --batch_size=4 \
            --seq_length=3 \
            --num_source=2 \
            --num_epochs=30 \
            --log_prefix=6

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode 1 \
            --batch_size=1 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.05 \
            --seq_length=3 \
            --num_source=2 \
            --dataset_dir /home/RAID1/DataSet/KITTI/KittiRaw/ \
            --output_dir ../test_output/test_depth \
            --log_prefix=6

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 9 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.05 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                 --log_prefix=6

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 10 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.0 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                --log_prefix=7

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
            --run_mode 1 \
            --batch_size=1 \
            --smooth_weight=0.5 \
            --explain_reg_weight=0.0 \
            --seq_length=3 \
            --num_source=2 \
            --dataset_dir /home/RAID1/DataSet/KITTI/KittiRaw/ \
            --output_dir ../test_output/test_depth \
            --log_prefix=7

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 9 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.0 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                 --log_prefix=7

/usr/local/anaconda3/bin/python3 ../Core/EvisionNet_MultiGPU.py \
                --run_mode 2 \
                --test_seq 10 \
                --batch_size 1 \
                --seq_length 3 \
                --seq_length=3 \
                --num_source=2 \
                --smooth_weight=0.5 \
                --explain_reg_weight=0.0 \
                --dataset_dir /home/RAID1/DataSet/KITTI/KittiOdometry/ \
                --log_prefix=7