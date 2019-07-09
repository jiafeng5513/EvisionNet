EvisonNet
=====
#### 使用无监督方法同时进行相机的标定,运动估计,深度估计
#### 可能的应用:算法防抖,基于互联网视频的三维重建.
## FBI WARNING!!
警告:谨慎参考和使用,未完成工作,BUG较多,随意使用有BOOM风险.
### 1. 目录说明<br>
* doc:文档和参考文献
* examples:测试数据
* MiddEval3:middlebury评价脚本
* MiddleburyScripts:读取Middlebury数据集的脚本
### 2. 环境说明<br>
* Windows10 or Ubuntu 18.04
* Python3.6 or 3.7,Anaconda3
* CUDA10+cudnn7.5
* tensorflow 1.13.1
### 3. 参考文献<br>
[1]. (双目视差)Chang, Jia-Ren, and Yong-Sheng Chen. "Pyramid stereo matching network." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.[PSMNet](https://github.com/JiaRenChang/PSMNet).<br>
[2]. (特征点匹配并应用于三维重建)Moo Yi K, Verdie Y, Fua P, et al. "Learning to assign orientations to feature points." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 107-116.[Code](https://github.com/vcg-uvic/benchmark-orientation).<br>
[3]. Verdie, Yannick, et al. "TILDE: a temporally invariant learned detector." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.[TILDE](https://github.com/cvlab-epfl/TILDE).<br>
[4]. Han, Xufeng, et al. "Matchnet: Unifying feature and metric learning for patch-based matching." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.[matchnet](https://github.com/hanxf/matchnet).<br>
[5]. Zhou, Tinghui, et al. "Unsupervised learning of depth and ego-motion from video." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.[SfMLearner](https://github.com/tinghuiz/SfMLearner).<br>
[6]. Yi, Kwang Moo, et al. "Lift: Learned invariant feature transform." European Conference on Computer Vision. Springer, Cham, 2016.<br>
[7]. Teng, Qianru, Yimin Chen, and Chen Huang. "Occlusion-Aware Unsupervised Learning of Monocular Depth, Optical Flow and Camera Pose with Geometric Constraints." Future Internet 10.10 (2018): 92.<br>
[8]. Liu, Qiang, et al. "Using Unsupervised Deep Learning Technique for Monocular Visual Odometry." IEEE Access 7 (2019): 18076-18088.<br>
[9]. DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras.[关键词:Camera Calibrate deep learning].<br>
[10]. Depth from Videos in the Wild:Unsupervised Monocular Depth Learning from Unknown Cameras.<br>
[11]. A Flexible New Technique for Camera Calibratio[张氏标定].<br>
### 4. 参考项目
[1]. [EdgeConnect](https://github.com/knazeri/edge-connect).<br>
[2]. [Cycled-GAN](https://github.com/andrea-pilzer/unsup-stereo-depthGAN/).<br>
[3]. [SfMLearner](https://github.com/tinghuiz/SfMLearner).<br>
[4]. [SfmLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch).<br>
### 5. 其他网址
[1]. [middlebury 数据集](http://vision.middlebury.edu/stereo/).<br>
[2]. [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/).<br>
[3]. [VIsion-SceneFlowDatasets数据集](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#faq).<br>
[3]. [Cycled-GAN解析](https://www.cnblogs.com/19991201xiao/p/9734422.html).<br>
[4]. [PSMNet解析](https://blog.csdn.net/zhiwei2coder/article/details/79929864?utm_source=blogxgwz3).<br>
[5]. [中科院自动化所三维重建数据集](http://vision.ia.ac.cn/zh/data/index.html).<br>
[6]. [SfMLearner(Depth and Ego-Motion)解析](https://zhuanlan.zhihu.com/p/50544334).<br>
[7]. [OpenMVS](https://github.com/cdcseacave/openMVS).<br>
[8]. [OpenMVG](https://github.com/openMVG/openMVG).<br>
[9]. [CVonline,图片数据集汇总](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm).<br>
[10]. [VisualData数据集搜索](https://www.visualdata.io/).<br>
[11]. [360D-zenodo Dataset]().<br>
[12]. [RGB-D Panorama Dataset](http://im2pano3d.cs.princeton.edu/).<br>
[13]. [Deep Depth Completion of a Single RGB-D Image解析](https://cloud.tencent.com/developer/news/322095).<br>
[14]. [Unsupervised Learning of Depth and Ego-Motion解析](https://zhuanlan.zhihu.com/p/50544334).<br>
[15]. [视觉里程计 第二部分：匹配、鲁棒、优化和应用](https://blog.csdn.net/cicibabe/article/details/70260936).<br>
[16]. [怎样通过照片获得高质量3D模型](https://zhuanlan.zhihu.com/p/24137374).<br>

### 6.性能记录
1. test_depth

| Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|---------|--------|-------|-----------|-------|-------|-------|
| 0.183   | 1.595  | 6.709 | 0.270     | 0.734 | 0.902 | 0.959 | 

2. test_pose

| Seq. 09            | Seq. 10            |
|--------------------|--------------------|
| 0.016 (std. 0.009) | 0.013 (std. 0.009) |

### 7.评价指标说明
1. 深度指标:

$abs\_rel=Mean(\left |\frac{gt-pred}{gt}\right|)$<br>
$sq\_rel=Mean(\frac{(gt-pred)^{2}}{gt})$ <br>
$rms=\sqrt{Mean((gt-pred)^{2})}$<br>
$log\_rms=\sqrt{Mean([(log(gt)-log(pred)]^{2})}$<br>
$a1=Mean((thresh<1.25))$<br>
$a2=Mean((thresh<1.25^{2}))$<br>
$a3=Mean((thresh<1.25^{3}))$<br>
$thresh=np.maximum((\frac{gt}{pred}), (\frac{pred}{ gt}))$<br>

2. ego-motion指标:
ATE(Absolute Trajectory Error,绝对轨迹误差)在测试集上的均值和标准差.<br>