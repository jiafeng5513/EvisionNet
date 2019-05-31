path = '../src/Utils/tools_evaluate/mex/';
distances = {'emd','l2','hamming','l1','sGLOH'}

for i=1:size(distances,2)
    distanceName = distances{i};
    eval(['mex ' path 'c_eoverlap.cxx ' path 'distance/' distanceName '.cxx  -outdir ' path])
    movefile([path 'c_eoverlap.' mexext],[path 'c_eoverlap_' distanceName '.' mexext]);
end