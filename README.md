EvisonNet
=====
### 1. 目录说明<br>
* doc:文档和参考文献
* examples:测试数据
* MiddEval3:middlebury评价脚本
* MiddleburyScripts:读取Middlebury数据集的脚本
### 2. 环境说明<br>
* Windows10 or Ubuntu 18.04
* Python3.6 or 3.7,Anaconda3
* CUDA10+cudnn7.5
* Pytorch
### 3. 参考文献<br>
[1]. (双目视差)Chang, Jia-Ren, and Yong-Sheng Chen. "Pyramid stereo matching network." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.[PSMNet](https://github.com/JiaRenChang/PSMNet).<br>
[2]. (特征点匹配并应用于三维重建)Moo Yi K, Verdie Y, Fua P, et al. "Learning to assign orientations to feature points." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 107-116.<br>
[3]. Verdie, Yannick, et al. "TILDE: a temporally invariant learned detector." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.[TILDE](https://github.com/cvlab-epfl/TILDE).<br>
[4]. Han, Xufeng, et al. "Matchnet: Unifying feature and metric learning for patch-based matching." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.[matchnet](https://github.com/hanxf/matchnet).<br>
[5]. Zhou, Tinghui, et al. "Unsupervised learning of depth and ego-motion from video." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.[SfMLearner](https://github.com/tinghuiz/SfMLearner).<br>
### 4. 参考项目
[1]. [EdgeConnect](https://github.com/knazeri/edge-connect)<br>
[2]. [Cycled-GAN](https://github.com/andrea-pilzer/unsup-stereo-depthGAN/)<br>
### 5. 其他网址
[1]. [middlebury 数据集](http://vision.middlebury.edu/stereo/)<br>
[2]. [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/)<br>
[3]. [VIsion-SceneFlowDatasets数据集](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#faq)<br>
[3]. [Cycled-GAN解析](https://www.cnblogs.com/19991201xiao/p/9734422.html)<br>
[4]. [PSMNet解析](https://blog.csdn.net/zhiwei2coder/article/details/79929864?utm_source=blogxgwz3)<br>
[5]. [中科院自动化所三维重建数据集](http://vision.ia.ac.cn/zh/data/index.html)<br>
[6]. [SfMLearner(Depth and Ego-Motion)解析](https://zhuanlan.zhihu.com/p/50544334)<br>
### 6.基本思路
1. 构造两个网络,一个是相机位姿估计网络,称为p-net,一个是特征点匹配网络,称为m-net.<br>
2. 两个网络共享一些层.<br>
3. p-net使用三联帧估计出相机的运动.<br>
4. m-net获取匹配点.<br>
5. 进行三维重建.<br>
6. 全部基于单目多视角图片序列.<br>
7. 参考文献2,3,4中的对于特征匹配的实验可以参考.<br>
8. SfMLearner的位置准确度实验可以参考.<br>
9. 参考文献2中的三维重建实验可以参考.<br>