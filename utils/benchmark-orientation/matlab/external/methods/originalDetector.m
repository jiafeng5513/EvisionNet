%%% originalDetector.m --- 
%% 
%% Filename: originalDetector.m
%% Description: 
%% Author: Kwang Moo Yi, Yannick Verdie
%% Maintainer: Kwang Moo Yi
%% Created: Wed Jun 29 13:44:22 2016 (-0700)
%% Version: 
%% Package-Requires: ()
%% URL: 
%% Doc URL: 
%% Keywords: 
%% Compatibility: 
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%%% Commentary: 
%% 
%% 
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%%% Change Log:
%% 
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright (C), EPFL Computer Vision Lab.
%% 
%%% Code:


function [] = originalDetector( methodName, in_file_name, out_file_name)
    global sRoot;

    if (~exist('p','var'))
        p = struct();
    end
    rootFolder = sRoot; 
    
    if isfield(p,'rootFolder')
        rootFolder = p.rootFolder;
    end

    [status, whereisOpenCV] = system('pkg-config opencv --libs-only-L');
    whereisOpenCV = whereisOpenCV(3:end-1);
    library_path_linux = sprintf([ 'LD_LIBRARY_PATH="' whereisOpenCV '" ; export LD_LIBRARY_PATH']);
    library_path_mac = sprintf([ 'DYLD_LIBRARY_PATH="' whereisOpenCV '" ; export DYLD_LIBRARY_PATH']);
    library_path = [library_path_mac ';' library_path_linux ';'];
    
    if (status ~= 0)
        error('whitout knowing where is opencv lib, I cannot do anything..., do you have pkg-config ?');
    end

    detector_path = [rootFolder '/external/methods/original' methodName 'detector'];
    com = sprintf('%s %s %s %s',library_path, detector_path,  in_file_name, out_file_name );

    val = system(com);
    if (val ~= 0)
        error('kp went wrong');
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% originalDetector.m ends here
