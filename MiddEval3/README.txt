MiddEvalSDK

Software development kit for the Middlebury Stereo Evaluation v.3

Version 1.6
Daniel Scharstein
September 14, 2015


=============================== Contents ==================================

README.txt          -- this file

CHANGES.txt         -- change log

COPYING_LESSER.txt, 
COPYING.txt         -- license (GNU Lesser General Public License)

alg-ELAS/           -- sample stereo algorithm: Libelas by A. Geiger,
                       adpated with permission from  
                       http://www.cvlibs.net/software/libelas/

alg-ELAS/run        -- sample run script (adapt this for your algorithm)

code/               -- C++ code used by the scripts

runalg              -- script to run algorithm on datasets
runeval             -- script to evaluate algorithm results
runevalF            -- script to evaluate algorithm results with full-size GT
runviz              -- script to visualize .pfm disparities as .png
makezip             -- script to create zip archive of results
cleanalg            -- script to remove files created by algorithms


In addition to the SDK files, you will also need to download input
data and ground truth data, and unzip at the same level as the SDK
files.  This should result in a subset of the following directories:

testF/
testH/
testQ/
trainingF/
trainingH/
trainingQ/


============================= Quick Start =================================


Follow these steps to get started.  We'll use the quarter-resolution
files for now.


1. Download and unzip MiddEval3-data-Q.zip and MiddEval3-GT0-Q.zip.
You should have a directory MiddEval3 with the following contents:

COPYING.txt         README.txt  cleanalg  makezip  runeval   runviz  testQ/
COPYING_LESSER.txt  alg-ELAS/   code/     runalg   runevalF  trainingQ/


2. Compile Libelas as follows:

cd alg-ELAS/build
cmake ..
make
cd ../..


3. Compile the tools in code/ as follows:

cd code/imageLib
make
cd ..
make
cd ..


4. Run ELAS:

./runalg                         -- shows usage
./runalg Q Motorcycle ELAS       -- run ELAS on one dataset
./runalg Q Motorcycle ELAS v2    -- run again, save results with suffix 'v2'
./runalg Q training ELAS         -- run ELAS on all 15 training sets


5. Evaluate results by ELAS:

./runeval                        -- shows usage
./runeval Q Motorcycle 1 ELAS    -- evaluate ELAS with error thresh t=1.0
./runeval Q Motorcycle 2 ELAS_s  -- evaluate "sparse" ELAS results, t=2.0
./runeval Q Motorcycle 1         -- evaluate all results for Motorcycle, t=1.0
./runeval Q training 0.5 ELAS    -- evaluate ELAS on all training sets, t=0.5

   *** NOTE: For efficiency, the evaluation script 'runeval' uses the
   GT file disp0GT.pfm with the SAME resolution as your disparity map.
   In contrast, the "official" online evaluation evaluates at FULL
   resolution, and upsamples your results if necessary.  Error
   thresholds need to be converted accordingly, e.g., a threshold of
   2.0 at full resolution would correspond to a threshold of 0.5 at
   quarter resolution.  Even with this conversion, the numbers differ
   slightly when evaluating with the GT at full resolution.  To get
   the official numbers, you can use the script 'runevalF'.  You'll
   need to download the full-size files MiddEval3-data-F.zip and
   MiddEval3-GT0-F.zip in order to use this script.  Once you do, try

./runevalF Q Motorcycle 1.0  ELAS   -- official numbers using full-res GT
./runeval  Q Motorcycle 0.25 ELAS   -- approximate numbers using Q-size GT

   In the online table for the training sets, the name of each method
   links to a zip file of the results.  You can download such a zip
   file, say results_SGM_Q.zip, unzip it *within* MiddEval3, and
   evaluate these results as well:

./runeval Q training 1 ELAS SGM

   Finally, to get evaluation output in column format ready to be imported
   into other programs such as Excel or gnuplot, use option -b ("brief"):

./runevalF -b Q all 2.0
./runeval  -b Q all 0.5


6. Examine disparity maps

   We highly recommend to install Heiko Hirschmueller's "cvkit" (link
   from webpage), which includes a powerful image viewer "sv".  Once
   installed, try the following:

sv trainingQ/Motorcycle/disp*pfm

   Hit 'h' for help, 'm' to change colormaps, 'k' to keep the colormap
   setting, and the arrow keys to switch between files.  Most amazing:
   hit 'p' to start the 3D viewer 'plyv' and visualize the disparities
   in 3D! (plyv reads the camera parameters from the calib.txt files.)

   Alternatively, run

./runviz Q

   to translate all disp*pfm files into disp*-c.png color encodings,
   and view those using any image viewer.  On subsequent runs, runviz
   will only re-convert any pfm files that have changed.


7. Add your own algorithm:

   Create a directory alg-XXX, where XXX is a (short) name for your
   method.  Then create a "run" script within your alg-XXX directory,
   using the one in alg-ELAS as a model.  Once done, test your
   algorithm.  For example, calling the new method "Ours":

./runalg Q Motorcycle Ours
./runeval Q Motorcycle 1 Ours

   A few words on the disparity encoding: We use floating-point
   disparities in PFM format, with INFINITY representing unknown
   values.  If your algorithm produces disparities in other formats
   (e.g., scaled 16-bit PNG or floating-point TIF), that's ok, and you
   can convert the output using the "imgcmd" tool (part of cvkit).
   Alternately, you can use the tool code/disp2pfm to convert integer
   disparities (e.g. in pgm format) to pfm.
   See the alg-ELAS/run script for more detailed information.  Note,
   however, that many of the half-size datasets and most of the
   full-size datasets have disparity ranges that exceed 255.  Thus, if
   your method produces 8-bit integer disparities, you are limited to
   working with the quarter-resolution datasets and won't be able to
   achieve accurate results when evaluated at full resolution.

   Alternately, you can change your implementation to write PFM
   disparities directly.  You can use the following code as examples:
     alg-ELAS/src/main.cpp
     code/imageLib/ImageIO.cpp


8. Submit your results:

   Our online web interface supports uploading of zip files of results
   on either just the 15 training images or all 30 training and test
   images.  You can upload the results on just the training set in
   order to see a temporary table comparing your results with those
   uploaded by others.  To do this, run

./makezip Q training Ours

   and upload the resulting file resultsQ-Ours.zip using the web
   interface.  This scripts ensures that all necessary files are
   present and that the runtimes files contain numbers.

   Once you are ready to submit your results to the permanent table,
   you must upload a complete set of results on ALL 30 datasets (15
   training, 15 test).  All images must have the same resolution (one
   of Q, H, or F).  You must use constant parameter settings (except
   for disaprity ranges), and you cannot "mix and match" resolutions
   (e.g., use trainingQ/Piano and trainingH/Teddy).  All of this is
   ensured if you use "runalg <res> all", e.g.:

./runalg Q all Ours

   Then, create a zip file with all results using

./makezip Q all Ours

   and upload the resulting file using the web interface.  You will
   then be able to provide information about your method and request
   publication of both training and test results.  Anonynous listings
   for double-blind review processes are possible.  Note that in order
   to prevent parameter tuning to the test data, you will have only
   one shot at evaluating your results on the test data.  In other
   words, you may not request publication for two different sets of
   results by the same method. Also, your results on the training data
   will be available for download by others.  So please be sure to
   only submit your final set of results for publication.  You may of
   course upload and evaluate results on the training set as often as
   you want.


9. To clean results created by algs (disp*pfm, time*txt, disp*-c.png), use:

./cleanalg                       -- shows usage
./cleanalg Q ELAS                -- removes all results by ELAS
./cleanalg Q ELASv2              -- removes all results by ELASv2
./cleanalg Q ELAS\*              -- removes all results by ELAS (all versions)
