%% setup_path.m --- 
% 
% Filename: setup_path.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:14:08 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:14:13 2015 (+0200)
%           By: Kwang
%     Update #: 1
% URL: 
% Doc URL: 
% Keywords: 
% Compatibility: 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Commentary: 
% 
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Change Log:
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (C), EPFL Computer Vision Lab.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Code:


global bSetupPathFin
global sRoot;

doSetupPath = false;
if ~exist('bSetupPathFin','var')
    doSetupPath = true;
else
    if(isempty(bSetupPathFin) || bSetupPathFin ~= true)
        doSetupPath = true;
    end
end
if doSetupPath
    bSetupPathFin = true;

addpath(sRoot);
addpath(genpath([sRoot '/src/KeypointDetectors']));
addpath(genpath([sRoot '/src/KeypointOrientations']));
addpath(genpath([sRoot '/src/KeypointDescriptors']));
addpath(genpath([sRoot '/src/Utils']));
addpath([sRoot '/filters']);

addpath([sRoot '/external/libs/vlfeat-0.9.18/toolbox/']);
addpath([sRoot '/external/libs/dollarToolbox']);
addpath(genpath([sRoot '/external/methods/']));

% if true || ~exist([sRoot '/external/src/OpenCVWrapper/opencvKeypointDetector'],'file')
%     disp('compiled files do not already exist, compiling...');
%     cd ../external;
%     try
%     eval(['!sh ' sRoot '/external/buildAll.sh clean'])    
%     eval(['!sh ' sRoot '/external/buildAll.sh'])
%     catch
%     cd ../src/;
%     end
%     cd ../src/;
% end

vl_setup;

%mkdir(sRoot,'resultAUCs');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% setup_path.m ends here
