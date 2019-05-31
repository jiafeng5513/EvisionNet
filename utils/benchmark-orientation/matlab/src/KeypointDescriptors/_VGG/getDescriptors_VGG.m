function [feat,desc,metric] = getDescriptors_VGG(img_info, kp_file_name, orientation_file_name, p)

    % here reject some kp
    listMethodsOK = {'SIFT','EdgeFociD','SIFTMult','EdgeFociDMult'};
    if (all(~strcmp(kp_file_name,listMethodsOK)))
        feat = [];
        desc = [];
        metric = '';
        return;
    end

    metric = 'L2';
    methodName = strsplit(mfilename,'_');
    methodName = methodName{end};

    param_nameKp =  p.optionalParametersKpName;
    param_nameOrient =  p.optionalParametersOrientName;
    param_nameDesc =  p.optionalParametersDescName;

    paramname = [param_nameKp '-' param_nameOrient '-' param_nameDesc];

    in = img_info.image_name;
    in = strrep(in, 'image_gray', 'image_color');

    kpf = [img_info.full_feature_prefix '_' kp_file_name '_keypoints_' orientation_file_name '_oriented-' param_nameKp '-' param_nameOrient '-txt'];
    out = [img_info.full_feature_prefix '_' kp_file_name '_keypoints_' orientation_file_name '_oriented_' methodName '_descriptors-' paramname '-txt'];


    kpf2 = [img_info.full_feature_prefix '_' kp_file_name '_keypoints_for' methodName '_' orientation_file_name '_oriented-' param_nameKp '-' param_nameOrient '-txt'];
    if ~exist(out, 'file')

        
        [feat_out, ~, ~] = loadFeatures(kpf);        
        saveFeatures(feat_out,kpf2);
        
        vgg_compute_function(in, kpf2, out, p);
        
    end
    if ~exist(kpf2, 'file')
        warning('for file does not exist, not using it');
    else
        kpf = kpf2;
    end    
    if ~exist(kpf, 'file')
        error('the keypoints do not exist, abort');
    end

    [feat, ~, ~] = loadFeatures(kpf);
    desc = loadDescriptors(out)';

    if (size(feat,2) ~= size(desc,2))
        display(com);
        error([methodName ' deleted kp, so now we have a missmatch !']);
    end

end

%  Copyright (c) 2014, Karen Simonyan
%  All rights reserved.
%  This code is made available under the terms of the BSD license (see COPYING file).

%% compute low-dimensional descriptors using the learnt pooling region and projection models

function [] = vgg_compute_function(in, kpf, out, p)
% This function compute low-dimensional descriptors using the learnt pooling region and projection models
    
    
    fRatioScale = 7.5;                  % we use SIFT settings for
                                        % extracting patch

    global sRoot;    
    if (~exist('p','var'))
        p = struct();
    end
    rootFolder = sRoot; 
    
    if isfield(p,'rootFolder')
        rootFolder = p.rootFolder;
    end
    
    % directory contatining vgg models
    rootDir = [rootFolder '/external/vgg_models'];

    % Upper bound on the projected descriptor dimensionality.
    % We simply used 64.
               
    % DescDim = 80;
    DescDim = 64;
    % DescDim = 48;

    % PR and projection models, learnt on different sets
    TrainData(1).Name = 'yosemite';
    TrainData(1).PRModel = 'PR_rank_m0.25_g4';

    TrainData(2).Name = 'notredame';
    TrainData(2).PRModel = 'PR_rank_m0.15_g0.5';

    TrainData(3).Name = 'liberty';
    TrainData(3).PRModel = 'PR_rank_m0.5_g8';

    switch DescDim

      case 80
        TrainData(1).ProjModel = 'proj_rank_m0.001_g0.5';
        TrainData(2).ProjModel = 'proj_rank_m0.001_g1';
        TrainData(3).ProjModel = 'proj_rank_m0.001_g1'; 

      case 64
        TrainData(1).ProjModel = 'proj_rank_m0.002_g0.5';
        TrainData(2).ProjModel = 'proj_rank_m0.001_g0.25';
        TrainData(3).ProjModel = 'proj_rank_m0.002_g1';

      case 48
        TrainData(1).ProjModel = 'proj_rank_m0.003_g0.25';
        TrainData(2).ProjModel = 'proj_rank_m0.003_g0.5';
        TrainData(3).ProjModel = 'proj_rank_m0.003_g0.5';

    end

    % training - evaluation set combinations
    TrainTestNames = [];

    TrainTestNames{end + 1} = {'yosemite', 'notredame'};
    TrainTestNames{end + 1} = {'yosemite', 'liberty'};
    TrainTestNames{end + 1} = {'yosemite', 'yosemite'};

    TrainTestNames{end + 1} = {'notredame', 'notredame'};
    TrainTestNames{end + 1} = {'notredame', 'yosemite'};
    TrainTestNames{end + 1} = {'notredame', 'liberty'};

    TrainTestNames{end + 1} = {'liberty', 'notredame'};
    TrainTestNames{end + 1} = {'liberty', 'liberty'};
    TrainTestNames{end + 1} = {'liberty', 'yosemite'};

    iTrain = 3; % number of the trainset we will use (we use liberty as it the only one not in our dataset)

    TrainSet = TrainData(iTrain).Name;    
    
    DataDir = '.';
    
    PRModelName = TrainData(iTrain).PRModel;
    ProjModelName = TrainData(iTrain).ProjModel;

    TrainDatasetDir = sprintf('%s/%s/%s/', rootDir, DataDir, TrainSet);
    
    % load image
    img = imread(in);
    
    % load features
    feat = loadFeatures(kpf);

    % load patches
    Patches = extract_patch_gray(img, feat, fRatioScale);
    
    % load PR filters
    PRFiltersPath = sprintf('%s/%s/common/PRFilters.mat', rootDir, DataDir);
    tmp = load(PRFiltersPath, 'PRFilters');
    PRFilters = tmp.PRFilters;
    
    ModelDir = sprintf('%s/models_PR/', TrainDatasetDir);
    PRModelPath = sprintf('%s/%s.mat', ModelDir, PRModelName);
    
    ModelData = load(PRModelPath, 'MinObj_w');
    w = ModelData.MinObj_w;
    
    % select pooling regions
    w = repmat(w', 8, 1);
    w = w(:);
    
    NZIdx = (w > 0) & any(PRFilters, 2);
    
    w = w(NZIdx);
    PRFilters = PRFilters(NZIdx, :);
    
    [PRFilters, ~, UniqueIdx2] = unique(PRFilters, 'rows');
    
    % load the projection model
    ModelDir = sprintf('%s/models_proj/', TrainDatasetDir);
    ProjModelPath = sprintf('%s/%s.mat', ModelDir, ProjModelName);
    
    ModelData = load(ProjModelPath, 'MinObj_W');
    Proj = ModelData.MinObj_W;
    
    % remove zero rows
    Proj = Proj(any(Proj,2), :);
    
    %% compute descriptors
    nPatches = size(Patches,3);
    
    % run the first patch to see desc dimension
    DescTmp = get_desc(squeeze(Patches(:,:,1)), PRFilters, 'Proj', Proj);
    Desc = zeros(size(DescTmp(:),1),nPatches);
    Desc(:, 1) = DescTmp(:);

    %     for iPatch = 1:nPatches
    for iPatch = 2:nPatches

        DescTmp = get_desc(squeeze(Patches(:,:,iPatch)), PRFilters, 'Proj', Proj);
        Desc(:, iPatch) = DescTmp(:);

    end
    
    %% save
    saveDescriptors(Desc',out);
    
    % end
end

function [p_w] = extract_patch_gray(img, feat, fRatioScale)
% Patch extraction for a given feature point in our format using VLFEAT

    gray_img = rgb2gray(img);

    patchR = 32;

    patch_size = 2*patchR+1;

    % turn the patches into Keypoint of VLFEAT
    frame = zeros(6,size(feat,2));
    frame(1:2,:) = feat(1:2,:);
    frame(3:6,:) = feat(10:13,:);

    % Use vl_covdet to extract patches
    % p_w = zeros(patch_size*patch_size,size(feat,2));
    [~,p_w] = vl_covdet(single(gray_img),'Frames',frame,'descriptor','Patch', ...
                        'EstimateAffineShape', false, 'EstimateOrientation', false, ...
                        'PatchResolution', patchR, 'PatchRelativeSmoothing', 0, ...
                        'PatchRelativeExtent', fRatioScale);

    % reshape the patches
    % disp(size(p_w));
    p_w_orig = reshape(p_w,[patch_size,patch_size,size(p_w,2)]);
    p_w_resize = imresize(p_w_orig,[64,64]);
    p_w = uint8(round(p_w_resize));

    % save('debug.mat','p_w','p_w_orig','p_w_resize','gray_img','img');

end
