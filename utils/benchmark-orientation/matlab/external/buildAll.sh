#!/bin/bash
folderToGoIn=("src/OpenCVWrapper/"      \
		  "src/OpenCVWrapper/"  \
		  "src/BRISK-master/"   \
		  "src/freak-master/"   \
		  "src/sGLOH/"          \
		  "src/mrogh/"          \
		  "src/kaze/"           \
		  "src/kaze/"           \
		  "src/libdaisy/"       \
		  "src/OpenCVFixedScaleWrapper/" );
nameMethods=( "opencvDescriptor"            \
		  "opencvDetector"          \
		  "originalBRISKdetector"   \
		  "originalFREAKdescriptor" \
		  "originalsGLOHdescriptor" \
		  "originalMROGHdescriptor" \
		  "originalKAZEdescriptor"  \
		  "originalKAZEdetector"    \
		  "originalDAISYdescriptor" \
		  "opencvFixedScaleDescriptor" );

root=$(pwd);
function finish {
    #   # Your cleanup code here
    cd $root
}
trap finish EXIT


# get length of an array
tLen=${#folderToGoIn[@]}
echo $tLen

# Read input
input=${1-"all"}

# Perform make or clean for each folder
for (( i=0; i<${tLen}; i++ )); do
    echo "Entering ${folderToGoIn[$i]}"
    cd ${folderToGoIn[$i]}
    echo "make $input"
    make $input
    cd $root
    echo "Fixing Symlinks"
    if [ "$input" == "clean" ]; then
	if [ -f "./methods/${nameMethods[$i]}" ]; then
	    rm ./methods/${nameMethods[$i]}
	fi
    else
	if ! [ -f "./methods/${nameMethods[$i]}" ]; then
	    ln -s ../${folderToGoIn[$i]}${nameMethods[$i]} ./methods/${nameMethods[$i]}
	fi
    fi    
done
