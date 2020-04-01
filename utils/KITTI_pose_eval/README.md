# KITTI odometry evaluation tool

## Usage

1. 每个轨迹文件`xx.txt`包含一个N x 12的矩阵, 其中 N 是测试帧数. 第i行通过3x4变换矩阵表示第i个姿态。
2. 3x4矩阵是旋转和平移的增广矩阵。
3. 评估一个或多个给定的轨迹，例如`data` 文件夹中的`09_pred.txt` 和 `10_pred.txt`  :
```shell script
    python evaluation.py 
            --result_dir=./data/   % 待评价的轨迹文件和输出结果所在目录
            --eva_seqs=09_pred,10_pred % 评价result_dir下的那些轨迹
            --notshow % 不显示绘图UI
            --pdf % 保存成pdf
            --png % 保存成图片
```
4. 评估 `data` 文件夹下的所有轨迹:
```shell script
  python evaluation.py --result_dir=./data/ --eva_seqs=* 
```
5.  评估结果存储于`./data/xx_eval/`. 

## Reference
<a href="http://www.cvlibs.net/datasets/kitti/eval_odometry.php" target="_blank">KITTI odometry development kit</a>
