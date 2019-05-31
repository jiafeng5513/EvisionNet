#!/bin/bash
# sGLOH keypoint matching usage example.
# Code by Fabio Bellavia (fbellavia@unipa.it),
# refer to: F. Bellavia, D. Tegolo, C. Valenti,
# "Keypoint descriptor matching with context-based orientation
# estimation", Image and Vision Computing 2014, and
# F. Bellavia, D. Tegolo, E. Trucco, "Improving SIFT-based Descriptors
# Stability to Rotations", ICPR 2010.
# Only for academic or other non-commercial purposes.
# Some code is partially adapted from K. Mikolajczyk:
# http://www.robots.ox.ac.uk/~vgg/research/affine/

# Files are compiled statically for linux x64 systems (in particular for
# Ubuntu 10.10 x64). Binaries for windows systems as well as x32 linux
# systems are also provided.

chmod a+x harrisz sgloh get_match distance *.sh

# compute HarrisZ keypoints (see other downloads)
./harrisz -i graf/graf1.bmp
./harrisz -i graf/graf3.bmp

# compute the sGLOH descriptor
# check the sgloh.ini file for further options
# NOTES:
# - to be sure that it works, image should be an uncompressed
#   24-bit bitmap with no ICC profile or color table - BMP version 3.
#   If you have ImagemMagick installed you can use:
#   convert <image> -type TrueColor  BMP3:<image without extension>.bmp
./sgloh graf/graf1.bmp graf/graf1_mk.txt graf/graf1_sgloh.txt
./sgloh graf/graf3.bmp graf/graf3_mk.txt graf/graf3_sgloh.txt

# compute the precision/recall
# check the distance.ini file for further options and sCOR and sGOR
./distance graf/graf1.bmp graf/graf3.bmp graf/graf1_sgloh.txt graf/graf3_sgloh.txt graf/H1to3p

# compute the distance matrix, and the nn and the nnr match index pairs
# see the file runme.m for how to import results in matlab,
# get_match.ini for the different options,including sCOr and sGOr
# WARNING:
# - the file get_match.ini used here is not the default included in
#   each platform subfolder 
./get_match graf/graf1_sgloh.txt graf/graf3_sgloh.txt graf/graf_nn.txt graf/graf_nnr.txt
mv ms graf/graf_ms

echo 'Launch the matlab script runme.m for further details...'
