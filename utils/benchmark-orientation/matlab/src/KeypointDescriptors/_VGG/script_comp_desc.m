%  Copyright (c) 2014, Karen Simonyan
%  All rights reserved.
%  This code is made available under the terms of the BSD license (see COPYING file).

%% compute low-dimensional descriptors using the learnt pooling region and projection models

clear;

run('./startup.m');

% upper bound on the projected descriptor dimensionality
% DescDim = 80;
% DescDim = 64;
DescDim = 48;

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

%% train-test combinations
for k = 1:numel(TrainTestNames)

    %% set paths & load data
    TrainSet = TrainTestNames{k}{1};    
    TestSet = TrainTestNames{k}{2};
    
    iTrain = strmatch(TrainSet, {TrainData.Name});
    PRModelName = TrainData(iTrain).PRModel;
    ProjModelName = TrainData(iTrain).ProjModel;

    TrainDatasetDir = sprintf('%s/%s/', DataDir, TrainSet);
    TestDatasetDir = sprintf('%s/%s/', DataDir, TestSet);
    
    % destination path
    DescDir = sprintf('%s/desc/train_%s/', TestDatasetDir, TrainSet);
    mkdir(DescDir);

    DescPath = sprintf('%s/desc_%d.mat', DescDir, DescDim);
    
    if exist(DescPath, 'file')
        continue;
    end

    % load patches
    PatchPath = sprintf('%s/patches/patches.mat', TestDatasetDir);
    load(PatchPath, 'Patches');
    
    % load PR filters
    PRFiltersPath = sprintf('%s/common/PRFilters.mat', DataDir);
    load(PRFiltersPath, 'PRFilters');
    
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
    nPatches = numel(Patches);
    
    Desc = single([]);

%     for iPatch = 1:nPatches
    parfor iPatch = 1:nPatches

        DescTmp = get_desc(Patches{iPatch}, PRFilters, 'Proj', Proj);

        Desc(:, iPatch) = DescTmp(:);

    end
    
    %% save
    save(DescPath, 'Desc');
    
end
