# Learning to Assign Orientations to Feature Points

This software is a MATLAB implemenation of the benchmark in [1]. This software is intended to be used in conjuction with the [learn-orientation](https://github.com/kmyid/learn-orientation) repository, i.e. they should be *cloned side-by-side*. By default, the software does *not* use GPU, but can be easily enabled by configuring Theano to do so.

In order to avoid computing the same thing multiple times, This software **USES CACHING BY DEFAULT**.  It will save computed keypoints and descriptors in sub-directory of the dataset directories (detail on the dataset section below). In case you need to reset them, be sure to erase the cache files.

This software is strictly for academic purposes only.  For other purposes, please contact us.  When using this software, please cite [1] and other appropriate publications if necessary (see matlab/external/licenses for details).

[1] K.  M.  Yi, Y.  Verdie, P.  Fua, and V.  Lepetit.  "Learning to Assign Orientations to Feature Poitns.", Computer Vision and Patern Recognition (CVPR), 2016 IEEE Conference on.


Contact:

<pre>
Kwang Moo Yi : kwang_dot_yi_at_epfl_dot_ch
Yannick Verdie : yannick_dot_verdie_at_epfl_dot_ch
</pre>

## Requirements

* MATLAB 2013b or higher (may run on older versions but not tested)
* Theano
* Numpy
* OpenCV (2.4.12+ or 3+)
* fftw3 (Daisy)
* libconfig (Daisy)
* wine (for running EdgeFoci software on Linux)
* ImageMagick (for the command 'convert') available both on Mac and Linux
* Pkg-config
* [VGG learned models from Oxford](http://www.robots.ox.ac.uk/~vgg/software/learn_desc/)
  - Download the `data_compute.tar` and copy the subfolders to the subfolders in `matlab/external/vgg_models`. `patches.mat` files are not needed.

### Important

Make sure the binaries provided in `matlab/external/methods` are compiled for your platform. If not, run `buildAll.sh` in `matlab/external`, `buildAll.m` in `matlab/external`, `buildAll.m` in `matlab/src/Utils/tools_evaluate/mex`

## Usage

At the `matlab/src` directory in MATLAB,

 ```matlab
 run_evaluate(<dataset_name>, <number_of_keypoints>)
 ```
 - `dataset_name`: Name of the dataset
 - `number_of_keypoints`: Maximum number of feature points per image. We use
   1000.

For example,
 ```matlab
 run_evaluate('Viewpoints', 1000);
 termDisp(0,'Oxford','','Viewpoints', 1000);
 ```
   
## About the models

Two models are released in this repository. One trained with Edge-Foci keypoints and SIFT descriptors, and the other trained with SIFT keypoints and SIFT descriptors. The former corresponds to EF-SIFT+, EF-Daisy-star, EF-VGG-star in the paper, which gave best performance. The latter corresponds to SIFT+.

Additionally, model trained with random rotation augmentations are provided as ef-sift-360 for convenience. This model should be used when severe rotations are expected.

## Directory Structure

<pre>
matlab : Main project directory
  |
  |------ src : Contains our main benchmark codes
  |
  |------ external : Contains  the 3rd party codes for compared methods

data : Dataset directory
</pre>

## Datasets

The dataset is available for download at the [project web page]( http://cvlab.epfl.ch/research/detect/orientation). Extract the archive to the corresponding data directory to use the datasets.

 *`Viewpoints` Dataset includes the following directories

  `chatnoir`, `duckhunt`, `mario`, `outside`, `posters`

 *`Webcam` Dataset includes the following directories:

  `Chamonix`, `Courbevoie`, `Frankfurt`, `Mexico`, `Panorama`, `StLouis`

 *`Oxford` and `EdgeFoci` Dataset includes the following directories:

  `bark`, `leuven`, `rushmore`, `wall`, `bikes`, `notredame`, `yosemite`, `boat`, `obama`, `trees`, `graf`, `paintedladies`, `ubc`
  
  Use only the subset `rushmore`, `notredame`, `yosemite`, `obama`, `paintedladies`, as the others are used for training.

 *`Strecha` Dataset consists of 2 sequences. The current release version of our benchmark software does not yet have the interface to convert the dataset into the form that can be used within the benchmark. We will release this one shortly.

 *`DTU` Dataset consists of 60 sequences. The current release version of our benchmark software does not yet have the interface to convert the dataset into the form that can be used within the benchmark.

For example, `chatnoir` directory should be located at `<project root>/data/Viewpoints/chatnoir`. Inside each dataset directory, the following directory structure should exist.

<pre>
Sequence Name
	  |------ test
		    |------ image_color
		    |------ image_gray
		    |------ homography
		    |------ features
		    |------ test_imgs.txt
		    |------ homography.txt
</pre>

* Note: features directory is automatically generated when running the benchmark software for caching

## Third Party Software

### Licenses

  Implementations in the `matlab/external` directory are mostly adaptations of 3rd party software into our evaluation framework.  For the terms of use for the 3rd party software, please refer to the license files in `matlab/external/licenses`


### Modifications

  For the VLFeat Library, we hacked into `vl/covdet.c` so that we gain control on the number of orientations the detectors return.  We set it to use a single orientation.

