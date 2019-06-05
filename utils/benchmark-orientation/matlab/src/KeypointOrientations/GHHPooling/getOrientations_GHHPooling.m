%%% getOrientations_GHHPooling.m --- 
%% 
%% Filename: getOrientations_GHHPooling.m
%% Description: 
%% Author: Kwang Moo Yi, Yannick Verdie
%% Maintainer: Yannick Verdie
%% Created: Thu Jul  7 11:25:58 2016 (+0200)
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


function [feat] = getOrientations_GHHPooling(img_info, kp_file_name, p)
    global sRoot;

    rootFolder = sRoot; 
    if isfield(p,'rootFolder')
        rootFolder = p.rootFolder;
    end
    
    
    param_nameKp =  p.optionalParametersKpName;
    param_nameOrient =  p.optionalParametersOrientName;
    name_method = strsplit(mfilename,'_');
    name_method = name_method{end};
    
    if isfield(p,'optionalParametersParamDir')
        parameters_dir_name = p.optionalParametersParamDir;
    else
        parameters_dir_name = 'parameters';
    end
    
    % here reject to run if we don't have param dir
    if (~exist([rootFolder, '/src/KeypointOrientations/', name_method, '/', parameters_dir_name], 'dir'))
        feat = [];
        % display(['not exist ', rootFolder, '/src/KeypointOrientations/', name_method, '/', parameters_dir_name]);
        return;        
    end

    %here reject some kp
    if (~(strcmp(kp_file_name,'SIFT') || strcmp(kp_file_name,'SIFTFIXED')) )
        feat = [];
        % display(['skipping orientation estimation for method ' kp_file_name']);
        return;
    end
    
    
    
    
    out = [img_info.full_feature_prefix, '_', kp_file_name, '_keypoints_', name_method, ...
           '_oriented-', param_nameKp, '-', param_nameOrient, '-txt'];   
    if ~exist(out, 'file')
        out2 = [img_info.full_feature_prefix, '_SIFT_keypoints_', name_method, ...
                '_oriented-', param_nameKp, '-', param_nameOrient, '-txt'];   
        if exist(out2, 'file') && strcmp(kp_file_name,'SIFTFIXED')
            copyfile(out2,out);
        else
            in_img = img_info.image_name;
            in_img = strrep(in_img, 'image_gray', 'image_color');
            in_kp = [img_info.full_feature_prefix '_' kp_file_name '_keypoints-' param_nameKp '-txt'];
            if ~exist(in_kp, 'file')
                in_kp
                error('the keypoints do not exist, abort');
            end

            config_file_full_name = [rootFolder, '/src/KeypointOrientations/', name_method, '/', parameters_dir_name, '/', ...
                                param_nameOrient, '.config'];

            %% Estimate Orientations Here
            %backup feature
            [feat_old, ~, ~] = loadFeatures(in_kp);

            retry_cnt = 0;            
            status = 1;
            while status ~= 0

                % Instead of moving to the python directory, launch a subshell which goes to
                % python directory and executes. Insert Cuda things here,
                % if you want to use cuda
                VENV = getenv('VIRTUAL_ENV');
                if ~strcmpi(VENV, '')
                    % Run the python script to test and get keypoints
                    prefix = ['LD_LIBRARY_PATH=' VENV '/lib/'];
                else
                    % Run the python script to test and get keypoints
                    prefix = ['LD_LIBRARY_PATH='];
                end
                com = ['(cd ' rootFolder '/../../learn-orientation-release/python-code/;' prefix '; OMP_NUM_THREADS=1 python runSingleTestWithFiles.py ' in_img ...
                       ' ' in_kp ' ' config_file_full_name ' ' out ')'];
                [status, result] = system(com);

                % Also, let's just use single thread...

                % If we have an error from python execution, retry five times and spit error as the error might just be from theano compile issues...
                if status ~= 0
                    retry_cnt = retry_cnt + 1;
                    if retry_cnt >= 5
                        display(result);
                        error('error in python execution!');
                    end
                end
            end
        end
    end 
    
    [feat, ~, ~] = loadFeatures(out);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% getOrientations_GHHPooling.m ends here
