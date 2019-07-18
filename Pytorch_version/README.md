## 1. What has been done

* Training has been tested on KITTI and CityScapes.
* Dataset preparation has been largely improved, and now stores image sequences in folders, making sure that movement is each time big enough between each frame
* 训练速度提升
* 改变序列长度的时候不需要重新进行数据准备
* You can still choose the former stacked frames dataset format.
* You can know compare with groud truth for your validation set. It is still possible to validate without, but you now can see that minimizing photometric error is not equivalent to optimizing depth map.
* 平滑损失应用于深度而不是视差(应用于视差,在权重0.5的时候并不容易收敛,参见tensorflow版本)
* 如此更改的平滑损失会提升深度预测效果,但是会使位置预测效果变差,可以在[here](train.py#L270)改成原来的情况.
* 下采样的时候权重除以2.3(原来是2),经验值

## 2.数据准备

### 2.1.[KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php): <br>
1. 下载脚本: [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip)<br>
2. 运行下面的脚本 <br>
3. `--with-depth` option will save resized copies of groundtruth to help you setting hyper parameters. <br>
4.  `--with-pose` will dump the sequence pose in the same format as Odometry dataset (see pose evaluation)<br>
```bash
python3 data/prepare_train_data.py \
        /path/to/raw/kitti/dataset/ \
        --dataset-format 'kitti' \
        --dump-root /path/to/resulting/formatted/data/ \
        --width 416 \
        --height 128 \
        --num-threads 4 \
        [--static-frames /path/to/static_frames.txt] \
        [--with-depth] \
        [--with-pose]
```
### 2.2.[Cityscapes](https://www.cityscapes-dataset.com/):
 1. 下载 `leftImg8bit_sequence_trainvaltest.zip 和 camera_trainvaltest.zip`. <br>
 2. 运行下面的脚本 <br>
```bash
python3 data/prepare_train_data.py \
        /path/to/cityscapes/dataset/ \
        --dataset-format 'cityscapes' \
        --dump-root /path/to/resulting/formatted/data/ \
        --width 416 \
        --height 171 \
        --num-threads 4
```
Notice that for Cityscapes the img_height is set to 171 because we crop out the bottom part of the image that contains the car logo, and the resulting image will have height 128.


## 3.训练
```bash
python3 train.py \
        /path/to/the/formatted/data/ \
        -b 4 \
        -m 0.2 \
        -s 0.1 \
        --epoch-size 3000 \
        --sequence-length 3 \
        --log-output \
        [--with-gt]
```
You can then start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
在KITTI上~30K iterations的训练可以得到较好的结果

## 4.评价与测试
### 4.1. 生成视差图<br>
```bash
python3 run_inference.py \
        --pretrained /path/to/dispnet \
        --dataset-dir /path/pictures/dir \
        --output-dir /path/to/output/dir
```
使用`dataset-dir`下的所有图片生成对应的深度图(或视差图)并存储在 `output-dir`.<br>
使用`-h`可以查看帮助.<br>
`--output-depth` for depth map<br>
`--output-disp` for disparity map<br>


### 4.2. 视差/深度预测功能的评价<br>
```bash
python3 test_disp.py \
        --pretrained-dispnet /path/to/dispnet \
        --pretrained-posenet /path/to/posenet \
        --dataset-dir /path/to/KITTI_raw \
        --dataset-list /path/to/test_files_list
```
test_file_list 在 kitti_eval文件夹下面. 
如果要和原论文的评价结果公平比较的话,不要给出posenet.
如果给出了posenet,它会被用来消除缩放因子(scale factor)的歧义,此时唯一用来获取缩放因子的GT将会是车速,这更接近于真实情况,但是显然测试效果会变差.

### 4.3. ego-motion功能的测评<br> 
Pose evaluation is also available on [Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Be sure to download both color images and pose !
```bash
python3 test_pose.py \
        /path/to/posenet \
        --dataset-dir /path/to/KITIT_odometry \
        --sequences [09]
```

**ATE** (*Absolute Trajectory Error*) is computed as long as **RE** for rotation (*Rotation Error*). 
**RE** between `R1` and `R2` is defined as the angle of `R1*R2^-1` when converted to axis/angle. 
It corresponds to `RE = arccos( (trace(R1 @ R2^-1) - 1) / 2)`.
While **ATE** is often said to be enough to trajectory estimation, **RE** seems important here as sequences are only `seq_length` frames long.

## 5.预训练网络
[Avalaible here](https://drive.google.com/drive/folders/1H1AFqSS8wr_YzwG2xWwAQHTfXN5Moxmx)
对应的训练参数:
```bash
python3 train.py \
        /path/to/the/formatted/data/ \
        -b4 \
        -m0 \
        -s2.0 \
        --epoch-size 1000 \
        --sequence-length 5 \
        --log-output \
        --with-gt
```

### 5.1 Depth Results

| Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|---------|--------|-------|-----------|-------|-------|-------|
| 0.181   | 1.341  | 6.236 | 0.262     | 0.733 | 0.901 | 0.964 | 

### 5.2 Pose Results

5-frames snippets used

|    | Seq. 09              | Seq. 10              |
|----|----------------------|----------------------|
|ATE | 0.0179 (std. 0.0110) | 0.0141 (std. 0.0115) |
|RE  | 0.0018 (std. 0.0009) | 0.0018 (std. 0.0011) | 


## 6.Discussion

Here I try to link the issues that I think raised interesting questions about scale factor, pose inference, and training hyperparameters

 - [Issue 48](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/48) : Why is target frame at the center of the sequence ?
 - [Issue 39](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/39) : Getting pose vector without the scale factor uncertainty
 - [Issue 46](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/46) : Is Interpolated groundtruth better than sparse groundtruth ?
 - [Issue 45](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/45) : How come the inverse warp is absolute and pose and depth are only relative ?
 - [Issue 32](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/32) : Discussion about validation set, and optimal batch size
 - [Issue 25](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/25) : Why filter out static frames ?
 - [Issue 24](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/24) : Filtering pixels out of the photometric loss
 - [Issue 60](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/60) : Inverse warp is only one way !

