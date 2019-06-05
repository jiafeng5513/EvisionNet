%%% run_evaluate.m --- 
%% 
%% Filename: run_evaluate.m
%% Description: 
%% Author: Kwang Moo Yi, Yannick Verdie
%% Maintainer: Kwang Moo Yi
%% Created: Tue Jun 28 14:10:11 2016 (-0700)
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


function [] = run_evaluate(datasetName,numKey,startSeq,endSeq)

    warning on backtrace

    addpath('Utils');
    global sRoot;
    tmp = mfilename('fullpath');tmp =  strsplit(tmp, '/');tmp= tmp(1:end-2);
    sRoot = strjoin(tmp,'/');
    setup_path

    % mean scale normalization applied for BRISK and ORB, as they have a
    % different justification for scale. We make them have about the same
    % scale value range as SIFT. I am bit unsure where we got this thing,
    % but it's from some paper if I recall correctly.
    parameters.meanScaleNormalisation = 4;
    parameters.methodsNeedingScaleNormalisation = {'BRISK','ORB'};
    
    % The name of the training set. This is appended in the result folder
    % name. It's quite useful if you train with different sets.
    parameters.models = {'Oxford'};

    idx_i = 1;
    parameters.testsets = cell(1,1);
    if strcmpi(datasetName,'Strecha')
        % Strecha
        parameters.testsets{idx_i} = 'Strecha/fountain'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'Strecha/herzjesu'; idx_i = idx_i + 1;
    elseif strcmpi(datasetName,'EdgeFoci')
        % EF
        parameters.testsets{idx_i} = 'EdgeFoci/notredame'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'EdgeFoci/paintedladies'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'EdgeFoci/rushmore'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'EdgeFoci/yosemite'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'EdgeFoci/obama'; idx_i = idx_i + 1;
    elseif strcmpi(datasetName,'Webcam')
        % Webcam
        parameters.testsets{idx_i} = 'Webcam/Chamonix'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'Webcam/Courbevoie'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'Webcam/Frankfurt'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'Webcam/Mexico'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'Webcam/Panorama'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'Webcam/StLouis'; idx_i = idx_i + 1;
    elseif strcmpi(datasetName,'Viewpoints')
        % Our Viewpoints dataset.
        parameters.testsets{idx_i} = 'Viewpoints/outside'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'Viewpoints/chatnoir'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'Viewpoints/duckhunt'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'Viewpoints/posters'; idx_i = idx_i + 1;
        parameters.testsets{idx_i} = 'Viewpoints/mario'; idx_i = idx_i + 1;
    elseif strcmpi(datasetName,'DTU')
        % DTU
        idxDTUValid = [1:60];
        for i = idxDTUValid(startSeq:endSeq) 
            parameters.testsets{idx_i} = ['DTU/scene' num2str(i)];
            idx_i = idx_i + 1;
        end
    else
        error('Unknown dataset!');
    end

    parameters.numberOfKeypoints = cell(1,length(parameters.testsets));
    for i = 1:length(parameters.testsets)
        parameters.numberOfKeypoints{i} = numKey;
    end


    % Thing to display as final result. Possible values are 
    possibleDisplay = {'BarGraph', 'MatchingScore','AUC_PrecisionRecall_NN', ...
                       'AUC_PrecisionRecall_threshold', ...
                       'AUC_PrecisionRecall_NNDRT','individual_PrecisionRecall_NN', ...
                       'individual_PrecisionRecall_threshold', ...
                       'individual_PrecisionRecall_NNDRT'};    
    parameters.whatToDisplay = {'AUC_PrecisionRecall_NN'};
    
    % for individual display, you can set the match with a key
    parameters.optionalMatchingLegend = 'img1 to img4';
    
    % The folder containing the list of parameters to test
    parameters.parameters_dir_name = 'prelearned';
    
    % Detectors to test
    parameters.bypassDisactivateDetector = {'EdgeFociD'};
    
    % Descriptors to test
    parameters.bypassDesactivateDescriptor = {'VGG', 'Daisy'};
    

    % Set this to true, if you want to delete ALL the cache. I personally
    % NEVER use this option and delete the cache folder manually
    cleanAll = 0;
    if cleanAll
        for i = 1:size(parameters.testsets,2) 
            where = [sRoot '/../data/' parameters.testsets{i} '/test'];
            system(['rm -r ' where '/features']);
        end
    end

    computeDescriptors(parameters);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% run_evaluate.m ends here
