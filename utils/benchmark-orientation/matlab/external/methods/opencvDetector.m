%%% opencvDetector.m --- 
%% 
%% Filename: opencvDetector.m
%% Description: 
%% Author: Kwang Moo Yi, Yannick Verdie
%% Maintainer: Kwang Moo Yi
%% Created: Wed Jun 29 13:44:48 2016 (-0700)
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


function [] = opencvDetector(method_name, in_file_name, out_file_name)
global sRoot;
       
    [status, whereisOpenCV] = system('pkg-config opencv --libs-only-L');
    whereisOpenCV = whereisOpenCV(3:end-1);
    library_path_linux = sprintf([ 'LD_LIBRARY_PATH="' whereisOpenCV '" ; export LD_LIBRARY_PATH']);
    library_path_mac = sprintf([ 'DYLD_LIBRARY_PATH="' whereisOpenCV '" ; export DYLD_LIBRARY_PATH']);
    library_path = [library_path_mac ';' library_path_linux ';'];
    
    if (status ~= 0)
        error('whitout knowing where is opencv lib, I cannot do anything..., do you have pkg-config ?');
    end

    detector_path = [sRoot '/external/methods/opencvDetector'];
    
    optionRemoveDuplicateOrientation = 0;
    if (strcmp(method_name,'SIFT'))
        optionRemoveDuplicateOrientation = 1;
    end
    if (strcmp(method_name,'SIFTMult'))
        method_name = 'SIFT'; % overwrite to SIFT if SIFTMult is requested
                              % Note that we run SIFT with
                              % optionRemoveDuplicateOrientation off
    end
    
    com = sprintf('%s %s %s %s %s %d',library_path, detector_path, method_name, in_file_name, out_file_name, optionRemoveDuplicateOrientation );

    val = system(com);
    if (val ~= 0)
        error('kp went wrong');
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% opencvDetector.m ends here
