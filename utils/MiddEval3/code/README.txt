code/README.txt

Daniel Scharstein, June 2014

This directory contains C++ tools as part of the "MiddEvalSDK" for the
Middlebury Stereo Evaluation v.3.

======================================================================
=== Compiling and usage

To compile:

cd imageLib
make
cd ..

make

This will produce the following executables:

======== evaldisp

Compute evaluation metrics (% of bad pixels and average error), called
by ../runeval script

  usage: ./evaldisp disp.pfm gtdisp.pfm badthresh rounddisp [mask.png]
    (or call with single argument badthresh to print column headers)

======== computemask

Compute non-occluded masks from pair of disparity maps, optionally
utilizing y-disparities for imperfect rectification.  This program was
used to generate the provided files mask0nocc.png and mask1nocc.png,
but is not used by the SDK scripts.

  usage: ./computemask disp0.pfm [disp0y.pfm] disp1.pfm dir mask.png [thresh=1.0]

  use dir=-1 for left->right, dir=1 for right->left

======== ii

(Short for "imageinfo"): a useful debuging tool that displays summary
information for PNG, PPM, and PFM files

  usage: ./ii [-b band] [-m] img1 [img2 ...]

  prints information about images given
  -a     : use absolute values
  -b B   : only print info about band B
  -m     : only print min max for each band, followed by filename

======== pfm2png

Converts float disparity maps to color images for visualization.  Called
by ../runviz script.

  usage: ./pfm2png [-c calib.txt] [-m dmin] [-d dmax] disp.pfm disp.png

  maps float disparities from dmin..dmax to colors using 'jet' color map
  unknown values INF are mapped to black
  dmin, dmax can be supplied or read from calib file,
  otherwise, min and max values of input data are used

======== png2pgm

Converts color images to gray.  Used by ../alg-ELAS/run to convert
color input images to pgm format.

  usage: ./png2pgm [-quiet] in out

======== disp2pfm

Converts integer disparity maps to pfm format.  Provided as an
alternate tool to imgcmd for researchers whose code produces integer
disparity maps.

  usage: ./disp2pfm disp.pgm disp.pfm [dispfact=1 [mapzero=0]]

  converts pgm/png disparities to pfm, optionally dividing by dispfact
  (use 2 for half-size, etc.)
  if mapzero=1, 0's are considered unknown disparities and mapped to
  infinity


======================================================================
=== License

All programs are provided "as is" under the terms of the GNU Lesser
General Public License and come WITHOUT ANY WARRANTY.  See
../COPYING.txt and ../COPYING_LESSER.txt for full license terms.  In
addition, the code in imageLib/ is covered under an Microsoft Research
Source Code License Agreement -- see imageLib/Copyright.h.
