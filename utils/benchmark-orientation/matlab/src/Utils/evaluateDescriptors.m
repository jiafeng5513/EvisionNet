%% evaluateDescriptors.m ---
%
% Filename: evaluateDescriptors.m
% Description:
% Author: Yannick Verdie, Kwang Moo Yi
% Maintainer: Yannick Verdie, Kwang Moo Yi
% Created: Tue Jun 16 17:13:51 2015 (+0200)
% Version:
% Package-Requires: ()
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


function [res] = evaluateDescriptors( trainset_name, testset_name, ...
                                      num_key, parameters)
    global sRoot;

    if isfield(parameters,'parameters_dir_name')
        parameters_dir_name = parameters.parameters_dir_name;
    else
        parameters_dir_name = 'parameters';
    end

    %for params tuning
    s = rng;%save state of rand
    if exist('stateRand.mat','file')
        s = load('stateRand.mat');
        s = s.s;
        rng(s);
    end


    %% get configuration =====================================================

    fprintf('Setting up configuration\n');
    nameFolder = testset_name;
    p.dataset_name  = nameFolder;
    p.trainset_name = trainset_name;
    p.testset_name = testset_name;
    p.omp_num_threads = '16';
    p.rootTest = [sRoot '/../data/' p.dataset_name '/test'];
    p.test_img_list_filename = fullfile(p.rootTest,'test_imgs.txt');

    setenv('OMP_NUM_THREADS', p.omp_num_threads);

    % end get config =========================================================


    %%prepare data =========================================================
    [imgs_list,imgs_no] = get_list(p.rootTest,p.test_img_list_filename);

    if (imgs_no < 1)
        errorR('WHAT ! no image...');
    end

    display('Importing Homographies...(Maps)');
    bHasHomography = false;
    if exist([p.rootTest '/homography.txt'], 'file') == 2
        lines = importdata([p.rootTest '/homography.txt']);
        bHasHomography = true;
        
        if (length(lines) ~= imgs_no/2)
            error(['Homography is provided but does not correspond ' ...
                   'to the right number of images']);
        end
    end

    display('Reading images...');
    Hs = cell(imgs_no/2,1);
    list_img_info = cell(imgs_no,1);
    for i_img = 1:imgs_no
        imgs{i_img} = imread(imgs_list{i_img});
        [pathstr,name,ext] = fileparts(imgs_list{i_img});
        idx_NN = strfind(pathstr,'/');pt = pathstr(1:idx_NN(end));
        imgs_c{i_img} = imread([pathstr(1:idx_NN(end)) 'image_color/' ...
                            name ext]);
        list_img_info{i_img}.image_gray = imgs{i_img};
        list_img_info{i_img}.image_color = imgs_c{i_img};
        list_img_info{i_img}.image_name = imgs_list{i_img};
        
        pathSaveFeatures = [pathstr(1:idx_NN(end)) 'features/'];
        if ~exist(pathSaveFeatures, 'dir')
            mkdir(pathSaveFeatures);
        end
        full_name = [pathSaveFeatures name];
        list_img_info{i_img}.image_gray = imgs{i_img};
        list_img_info{i_img}.full_feature_prefix = full_name;
        list_img_info{i_img}.name = name;
        
        if (i_img<=imgs_no/2)%load only half (pair)
            Hs{i_img} = eye(3);
            
            %has custom homography provided. Actually, this is the case when
            %we use point-to-point mapping with 3D models for the Strecha
            %dataset and the DTU dataset
            if (bHasHomography)
                if ~exist([p.rootTest '/' lines{i_img}],'file')
                    error(['file ' p.rootTest '/' lines{i_img} ' ' ...
                                        'does not exist!']);
                end
                
                Hs{i_img} = importdata([p.rootTest '/' lines{i_img}]);
            end
        end
    end



    %%%%%here is how to compute the nb of keypoints
    % Note that this is just from the TILDE paper's style. Just for future
    % extensions, but should not be used (hence the error message)
    if (num_key < 1)
        error(['This part of the code was developed using hack just to work ' ...
               'for the case of the TILDE paper. Do not use this part unless ' ...
               'you know absolutely what you are doing!']);
        radiusHardcoded  = 5;
        magic_number = 10/8; % should make random's repeatability to the
                             % desired stuff
        imgArea = size(imgs_c{1},1)*size(imgs_c{1},2);
        num_key = magic_number * num_key * (imgArea) / (pi * ...
                                                        radiusHardcoded ...
                                                        * radiusHardcoded);
        num_key = uint16(num_key);
        disp(['-------------->' trainset_name '_run_on_' testset_name ...
              'Comparison_' num2str(num_key)]);
        
    end


    display('Reading list of methods and parameters to evaluate...');
    %get the name of the method we are going to test....
    %method starting with _ are ignored
    list_methodKp = dir('KeypointDetectors');
    isub = [list_methodKp(:).isdir]; %# returns logical vector
    list_methodKp = {list_methodKp(isub).name}';
    list_methodKp(ismember(list_methodKp,{'.','..'})) = [];
    idx_NN = cellfun(@(x) x(1) == '_', list_methodKp);
    %%%%%%%%%%%%% bypassDisactivateDetector
    if (isfield(parameters,'bypassDisactivateDetector'))
        idxbypass = false(1,size(idx_NN,1));
        for i=1:size(parameters.bypassDisactivateDetector,2)
            idxbypass = idxbypass | strcmpi(list_methodKp',['_', ...
                                parameters.bypassDisactivateDetector{i}]);
        end
        idx_NN = idx_NN & ~idxbypass';
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    list_methodKp(idx_NN) = [];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%remove underscore for the method bypassed
    for i=1:size(list_methodKp,1)
        n = list_methodKp{i};
        if (n(1) == '_')
            list_methodKp{i} = n(2:end);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nNumMethodsKp = size(list_methodKp,1);

    if (nNumMethodsKp == 0 )
        error('need kp method , error !')
    end

    %get the parameters
    [list_parametersKp, list_parameters_name_Kp] = ...
        getParameters('KeypointDetectors', list_methodKp, parameters_dir_name);
    nNumParamKp = size(list_parameters_name_Kp,2);


    %get the name of the method we are going to test....
    %method starting with _ are ignored
    list_methodOrient = dir('KeypointOrientations');
    isub = [list_methodOrient(:).isdir]; %# returns logical vector
    list_methodOrient = {list_methodOrient(isub).name}';
    list_methodOrient(ismember(list_methodOrient,{'.','..'})) = [];
    idx_NN = cellfun(@(x) x(1) == '_', list_methodOrient);
    list_methodOrient(idx_NN) = [];

    nNumMethodsOrient = size(list_methodOrient,1);

    if (nNumMethodsOrient == 0)
        error('no Orientation method, error !')
    end

    %get the parameters
    [list_parametersOrient, list_parameters_name_Orient] = ...
        getParameters('KeypointOrientations', list_methodOrient, ...
                                    parameters_dir_name);
    nNumParamOrient = size(list_parameters_name_Orient,2);







    %get the name of the method we are going to test....
    %method starting with _ are ignored
    list_methodDesc = dir('KeypointDescriptors');
    isub = [list_methodDesc(:).isdir]; %# returns logical vector
    list_methodDesc = {list_methodDesc(isub).name}';
    list_methodDesc(ismember(list_methodDesc,{'.','..'})) = [];

    idx_NN = cellfun(@(x) x(1) == '_', list_methodDesc);

    %%%%%%%%%%%%% bypassDesactivateDescriptor
    if (isfield(parameters,'bypassDesactivateDescriptor'))
        idxbypass = false(1,size(idx_NN,1));
        for i=1:size(parameters.bypassDesactivateDescriptor,2)
            idxbypass = idxbypass | strcmpi(list_methodDesc',['_', ...
                                parameters.bypassDesactivateDescriptor{i}]);
        end
        idx_NN = idx_NN & ~idxbypass';
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    list_methodDesc(idx_NN) = [];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%remove underscore for the method bupassed
    for i=1:size(list_methodDesc,1)
        n = list_methodDesc{i};
        if (n(1) == '_')
            list_methodDesc{i} = n(2:end);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nNumMethodsDesc = size(list_methodDesc,1);

    if (nNumMethodsDesc == 0)
        error('no Desc method, error !')
    end

    %get the parameters
    [list_parametersDesc, list_parameters_name_Desc] = ...
        getParameters('KeypointDescriptors', list_methodDesc, ...
                                    parameters_dir_name);
    nNumParamDesc = size(list_parameters_name_Desc,2);


    display('Preallocating Cells...');
    % Preallocate cells according to maximum size
    nbMaxComb = imgs_no / 2 * nNumMethodsKp * nNumParamKp * ...
        nNumMethodsOrient * nNumParamOrient * nNumMethodsDesc * nNumParamDesc;
    parametersUnrolled = cell(1,nbMaxComb);
    descriptor = cell(nbMaxComb,imgs_no);
    features = cell(nbMaxComb,imgs_no);

    parametersUnrolledidx = 1;

    display('Extracting/Loading Descriptors...');
    % detect keypoints for all images
    for i_img = 1:imgs_no
        
        %for all kp
        methodidx = 1;
        for i_methodKp = 1:nNumMethodsKp
            keyptMethodName = list_methodKp{i_methodKp};
            
            %all parameter of kp
            for i_paramKp = 1:nNumParamKp
                
                param = list_parametersKp{i_methodKp, i_paramKp};
                p.optionalParametersKp = param;
                p.optionalParametersKpName = ...
                    list_parameters_name_Kp{i_methodKp, i_paramKp};
                p.optionalParametersParamDir = parameters_dir_name;
                
                if  isempty(list_parameters_name_Kp{i_methodKp, i_paramKp})
                    % display(' - - Skip!');
                    continue;
                end
                
                eval(['getKeypoints_' keyptMethodName '(list_img_info{i_img},p);']);
                
                res_desc = cell(nNumMethodsOrient, nNumParamOrient, ...
                                nNumMethodsDesc, nNumParamDesc);
                res_feat = cell(nNumMethodsOrient, nNumParamOrient, ...
                                nNumMethodsDesc, nNumParamDesc);
                parametersRolled = cell(nNumMethodsOrient, ...
                                        nNumParamOrient, nNumMethodsDesc, ...
                                        nNumParamDesc);
                
                p.rootFolder = sRoot;%%%with parfor, global are not propagated, so paassing the path that way is the solution...
                
                for i_methodOrient = 1:nNumMethodsOrient
                    orientMethodName = list_methodOrient{i_methodOrient};
                    %%%%parallel for here
                    for i_paramorient = 1:nNumParamOrient
                        localp = struct(p);
                        
                        param = list_parametersOrient{i_methodOrient, ...
                                            i_paramorient};
                        localp.optionalParametersOrient = param;
                        localp.optionalParametersOrientName = ...
                            list_parameters_name_Orient{i_methodOrient, ...
                                            i_paramorient};
                        localp.optionalParametersParamDir = parameters_dir_name;
                        
                        if  isempty(localp.optionalParametersOrientName)
                            % display(' - - Skip!');
                            continue;
                        end
                        
                        functionHandle1 =str2func(['getOrientations_' ...
                                            orientMethodName ]);
                        f = functionHandle1(list_img_info{i_img}, ...
                                            keyptMethodName, localp);
                        
                        if  isempty(f)%this configuration kp->orientation is blocked, skip.
                                      % display(' - - Skip!');
                            continue;
                        end
                        
                        %for all Desc
                        for i_methodDesc = 1:nNumMethodsDesc
                            descMethodName = list_methodDesc{i_methodDesc};
                            
                            localp.optionalParameters = [];%just to be sure, not really needed
                            for i_paramdesc = 1:nNumParamDesc
                                                                
                                param = list_parametersDesc{i_methodDesc,i_paramdesc};
                                localp.optionalParametersDesc = param;
                                localp.optionalParametersDescName = list_parameters_name_Desc{i_methodDesc,i_paramdesc};
                                
                                if  isempty(list_parameters_name_Desc{i_methodDesc,i_paramdesc})
                                    % display(' - - Skip!');
                                    continue;
                                end
                                
                                functionHandle2 =str2func(['getDescriptors_' descMethodName]);
                                [feat,desc,metric] = functionHandle2(list_img_info{i_img},keyptMethodName,orientMethodName,localp);
                                
                                if  isempty(desc)%this configuration kp->desc is blocked, skip.
                                                 %display([' - - Skip!' keyptMethodName ' ' descMethodName list_parameters_name_Desc{i_methodDesc,i_paramdesc}]);
                                    continue;
                                end
                                
                                if (size(feat,2) ~= size(desc,2))
                                    error('missmatch size feature and descriptors !');
                                end
                                %
                                %here limit feature and descriptors..... based on num_key
                                
                                % We do special treatement for multi-orientation kp
                                if isempty(strfind(keyptMethodName,'Mult'))
                                    desc = desc(:,1:min(size(desc,2),num_key));
                                    feat = feat(:,1:min(size(feat,2),num_key));
                                else
                                    % find number of unique elements based on x,y,scale
                                    [~,~,ic] = unique(feat(1:3,:)','rows','stable');
                                    num_unique = max(ic);
                                    num_2_keep = min(num_unique,num_key);
                                    idx_2_keep = ic <= num_2_keep;
                                    last_idx_2_keep = max(find(ic == num_2_keep));
                                    
                                    if (last_idx_2_keep ~= sum(idx_2_keep))
                                        error('the repeated features seem to be not consecutive!');
                                    end
                                    
                                    desc = desc(:,idx_2_keep);
                                    feat = feat(:,idx_2_keep);
                                end
                                
                                %
                                res_desc{i_methodOrient,i_paramorient,i_methodDesc,i_paramdesc} = desc;
                                res_feat{i_methodOrient,i_paramorient,i_methodDesc,i_paramdesc} = feat;
                                
                                par = struct();
                                par.i_img = i_img;
                                par.i_img2 = i_img + imgs_no/2;
                                par.methodidx = methodidx;%%is not right here, will be updated in the seq. for loop
                                
                                par.i_methodKp = i_methodKp;
                                par.i_paramkp = i_paramKp;
                                par.keyptMethodName = keyptMethodName;
                                par.optionalParametersKp = localp.optionalParametersKp;
                                par.optionalParametersKpName = localp.optionalParametersKpName;
                                
                                par.i_methodOrient = i_methodOrient;
                                par.i_paramorient = i_paramorient;
                                par.orientMethodName = orientMethodName;
                                par.optionalParametersOrient = localp.optionalParametersOrient;
                                par.optionalParametersOrientName = localp.optionalParametersOrientName;
                                
                                par.i_methodDesc = i_methodDesc;
                                par.i_paramdesc = i_paramdesc;
                                par.descMethodName = descMethodName;
                                par.optionalParametersDesc = localp.optionalParametersDesc;
                                par.optionalParametersDescName = localp.optionalParametersDescName;
                                par.descMetric = metric;
                                
                                parametersRolled{i_methodOrient,i_paramorient,i_methodDesc,i_paramdesc} = par;
                                
                            end
                        end
                    end
                end
                
                %disp(num2str(methodidx))
                %%%%%%format the parallel results
                for i_methodOrient = 1:nNumMethodsOrient
                    % if  isempty(res_desc{i_methodOrient})
                    %     continue;
                    % end
                    for i_paramorient = 1:nNumParamOrient
                        % if  isempty(res_desc{i_methodOrient}{i_paramorient})
                        %     continue;
                        % end
                        
                        %for all Desc
                        for i_methodDesc = 1:nNumMethodsDesc
                            % if  isempty(res_desc{i_methodOrient}{i_paramorient}{i_methodDesc})
                            %     continue;
                            % end
                            
                            for i_paramdesc = 1:nNumParamDesc
                                
                                desc    = res_desc{i_methodOrient,i_paramorient,i_methodDesc,i_paramdesc};
                                feat    = res_feat{i_methodOrient,i_paramorient,i_methodDesc,i_paramdesc};
                                par     = parametersRolled{i_methodOrient,i_paramorient,i_methodDesc,i_paramdesc};
                                par.methodidx = methodidx;
                                
                                if  isempty(desc)
                                    continue;
                                end
                                
                                %and save
                                descriptor{methodidx,i_img} = desc;
                                features{methodidx,i_img} = feat;
                                
                                %unroll the loop for parallel for later...
                                if (i_img <= imgs_no/2)
                                    parametersUnrolled{1,parametersUnrolledidx} = par;
                                    parametersUnrolledidx = parametersUnrolledidx+1;
                                end
                                display(['img (' num2str(i_img) '/' num2str(imgs_no) ') - Loaded data for method ' num2str(methodidx)]);
                                methodidx = methodidx+1;
                            end
                        end
                    end
                end
                
                
            end
        end
    end

    % Remove Unneeded from preallocation
    parametersUnrolled = parametersUnrolled(1,1:parametersUnrolledidx-1);

    descriptor = descriptor(1:methodidx-1,:);
    features = features(1:methodidx-1,:);
    %get matching score
    matchingScore = zeros(1,(methodidx-1)*imgs_no/2);
    matchingScore_80 = zeros(1,(methodidx-1)*imgs_no/2);
    legend2_str = cell(1,(methodidx-1)*imgs_no/2);
    legend1_str = cell(1,(methodidx-1)*imgs_no/2);
    legend0_str = cell(1,(methodidx-1)*imgs_no/2);
    repeatability_NN = zeros(1,(methodidx-1)*imgs_no/2);
    precision_NN = cell(1,(methodidx-1)*imgs_no/2);
    recall_NN = cell(1,(methodidx-1)*imgs_no/2);
    precision_NNDRT = cell(1,(methodidx-1)*imgs_no/2);
    recall_NNDRT = cell(1,(methodidx-1)*imgs_no/2);
    precision_NNDRT_80 = cell(1,(methodidx-1)*imgs_no/2);
    recall_NNDRT_80 = cell(1,(methodidx-1)*imgs_no/2);
    precision_threshold = cell(1,(methodidx-1)*imgs_no/2);
    recall_threshold = cell(1,(methodidx-1)*imgs_no/2);


    %% The parfor loop to compute descriptor performances
    display('Evaluating Descriptors...');
    warning('parfor removed')
    for ipar=1:size(parametersUnrolled,2)
        
        %         par.i_img = i_img;
        par = parametersUnrolled{1,ipar};
        methodidx = par.methodidx;
        i_img = par.i_img;
        i_img2 = par.i_img2;
        
        keyptMethodName = par.keyptMethodName;
        i_methodKp = par.i_methodKp;
        i_paramKp = par.i_paramkp;
        
        orientMethodName = par.orientMethodName;
        i_methodOrient = par.i_methodOrient;
        i_paramorient = par.i_paramorient;
        
        descMethodName = par.descMethodName;
        i_methodDesc = par.i_methodDesc;
        i_paramdesc = par.i_paramdesc;
        descMetric = par.descMetric;
        
        
        a1 = features{methodidx,i_img}(7,:);
        b1 = features{methodidx,i_img}(8,:);
        c1 = features{methodidx,i_img}(9,:);
        
        a2 = features{methodidx,i_img2}(7,:);
        b2 = features{methodidx,i_img2}(8,:);
        c2 = features{methodidx,i_img2}(9,:);
        
        %        BRISK and other who want normalized scale for evaluation
        if isfield(parameters,'meanScaleNormalisation') && ~isempty(cell2mat(strfind(parameters.methodsNeedingScaleNormalisation,keyptMethodName)))
            coeff = mean(features{methodidx,i_img}(3,:))/parameters.meanScaleNormalisation;
            a1 = a1 * coeff * coeff;
            b1 = b1 * coeff * coeff;
            c1 = c1 * coeff * coeff;
            
            coeff = mean(features{methodidx,i_img2}(3,:))/parameters.meanScaleNormalisation;
            a2 = a2 * coeff * coeff;
            b2 = b2 * coeff * coeff;
            c2 = c2 * coeff * coeff;
        end
        
        f1abc = ([features{methodidx,i_img}(1:2,:);a1;b1;c1]);%%pi*features{i_methodkp}{i_img}(4,:)/180.
        f2abc = ([features{methodidx,i_img2}(1:2,:);a2;b2;c2]);%
                                                               %                     f2abcd = vl_frame2oell([features{i_methodkp}{i_img2}(1:2,:);features{i_methodkp}{i_img2}(6,:);pi*features{i_methodkp}{i_img2}(4,:)/180.]);
        f1desc =  descriptor{methodidx,i_img};
        f2desc =  descriptor{methodidx,i_img2};
        
        f1 = [f1abc;f1desc];
        f2 = [f2abc;f2desc];
        
        H =  Hs{i_img};
        
        % VLFEAT Descriptor Evaluation        
        legend0 = [ keyptMethodName '(' list_parameters_name_Kp{i_methodKp,i_paramKp} ') ' ...
                    orientMethodName '(' list_parameters_name_Orient{i_methodOrient,i_paramorient} ') ' ...
                    descMethodName '(' list_parameters_name_Desc{i_methodDesc,i_paramdesc} ') '];
        
        legend1 =  [ '(' num2str(size(f1,2)) ',' num2str(size(f2,2)) ')' ...
                     legend0 ...
                     ' -> ' list_img_info{i_img}.name ' to ' list_img_info{i_img2}.name ];
        
        legend2 =  [list_parameters_name_Kp{i_methodKp,i_paramKp}             '-'...
                    list_parameters_name_Orient{i_methodOrient,i_paramorient} '-'...
                    list_parameters_name_Desc{i_methodDesc,i_paramdesc}             ];
        
        if isfield(parameters, 'optionalFinalSuffix')            
            currentName = ['FinalDump_' num2str(size(f1,2)) '_' ...
                           num2str(size(f2,2)) '_' legend1 '_' ...
                           parameters.optionalFinalSuffix ];
        else            
            currentName = ['FinalDump_' num2str(size(f1,2)) '_' ...
                           num2str(size(f2,2)) '_' legend1 ];
        end
        
        namefile = [pathSaveFeatures currentName '.mat'];
        
        %%either load or compute....
        if exist(namefile,'file')
            disp(['loading ' namefile]);
            loadkey = load(namefile);
            matchingScore(1,ipar) = loadkey.match_score;
            matchingScore_80(1,ipar) = loadkey.match_score_80;
            legend0_str{1,ipar} = loadkey.legend0;
            legend1_str{1,ipar} = loadkey.legend1;
            legend2_str{1,ipar} = loadkey.legend2;
            repeatability_NN(1,ipar) = loadkey.repeatability_nn_cur;
            precision_NN{1,ipar} = loadkey.precision_t_NN;
            recall_NN{1,ipar} = loadkey.recall_t_NN;
            precision_threshold{1,ipar} = loadkey.precision_t_threshold;
            recall_threshold{1,ipar} = loadkey.recall_t_threshold;
            precision_NNDRT{1,ipar} = loadkey.precision_t_NNDRT;
            recall_NNDRT{1,ipar} = loadkey.recall_t_NNDRT;
            precision_NNDRT_80{1,ipar} = loadkey.precision_t_NNDRT_80;
            recall_NNDRT_80{1,ipar} = loadkey.recall_t_NNDRT_80;
            continue;
        end
        
        disp(['computing for ' namefile]);
        
        
        [~,~,~, match_score,~, twi0, twi1, dout0, tdout0, ~, ~,sdout0]= ...
            repeatability_noLoadFile(f1,f2,H, imgs{i_img}, imgs{i_img2},descMetric);        
        
        %%%%%%%%%%%%%%%%%%%%%%
        repeatability_nn_cur = sum(twi1(:)) / size(f1,2); % compute
                                                          % repeatability
                                                          % as well
        display(['repeatability = ' num2str(repeatability_nn_cur)]);
        %%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%%%%
        [precision_t_threshold, recall_t_threshold] = getPrecisionRecallTheshold(twi0, tdout0, [legend1 ': (Threshold)']);
        %%%%%%%%%%%%%%%%%%%%%%
        % %first half is all the points, then second half equally spaced
        % select = [round(1:size(precision_t_threshold,2)/(10000-1):size(precision_t_threshold,2)),size(precision_t_threshold,2)];
        % recall_t_threshold = recall_t_threshold(1,select);
        % precision_t_threshold = precision_t_threshold(1,select);
        
        %%%%%%%%%%%%%%%%%%%%%%
        [precision_t_NN, recall_t_NN] = getPrecisionRecall(twi0,twi1, dout0, [legend1 ': (NN)']);
        %%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%
        [precision_t_NNDRT, recall_t_NNDRT] = getPrecisionRecall(twi0,twi1, sdout0, [legend1 ': (NNDRT)'], 'descend');
        %%%%%%%%%%%%%%%%%%%%%%
        
        %%block 2nn/1nn <= 1/0.8 (1nn/2nn > 0.8)
        sdout0_80   = sdout0;
        sdout0_80(sdout0_80 <= 1/0.8) = 0;
        dx=(sdout0_80>0).*(twi1);
        matches_80=sum(sum(dx));
        match_score_80=100*matches_80/min(size(sdout0,1),size(sdout0,2));
        
        %%%%%%%%%%%%%%%%%%%%%%
        [precision_t_NNDRT_80, recall_t_NNDRT_80] = getPrecisionRecall(twi0,twi1, sdout0_80, [legend1 ': (NNDRT 80%)'], 'descend');
        %%%%%%%%%%%%%%%%%%%%%%
        
        if strcmpi(list_img_info{i_img}.name,'img1') && strcmpi(list_img_info{i_img2}.name,'img4')
            display('here!');
        end
        
        matchingScore(1,ipar) = match_score;
        matchingScore_80(1,ipar) = match_score_80;
        legend0_str{1,ipar} = legend0;
        legend1_str{1,ipar} = legend1;
        legend2_str{1,ipar} = legend2;
        repeatability_NN(1,ipar) = repeatability_nn_cur;
        precision_NN{1,ipar} = precision_t_NN;
        recall_NN{1,ipar} = recall_t_NN;
        precision_threshold{1,ipar} = precision_t_threshold;
        recall_threshold{1,ipar} = recall_t_threshold;
        precision_NNDRT{1,ipar} = precision_t_NNDRT;
        recall_NNDRT{1,ipar} = recall_t_NNDRT;
        precision_NNDRT_80{1,ipar} = precision_t_NNDRT_80;
        recall_NNDRT_80{1,ipar} = recall_t_NNDRT_80;
        
        toSave = struct();
        toSave.match_score = match_score;
        toSave.match_score_80 = match_score_80;
        toSave.legend0 =legend0;
        toSave.legend1 =legend1;
        toSave.legend2 = legend2;
        toSave.repeatability_nn_cur = repeatability_nn_cur;
        toSave.precision_t_NN = precision_t_NN;
        toSave.recall_t_NN = recall_t_NN;
        toSave.precision_t_threshold = precision_t_threshold;
        toSave.recall_t_threshold = recall_t_threshold;
        toSave.precision_t_NNDRT = precision_t_NNDRT;
        toSave.recall_t_NNDRT = recall_t_NNDRT;
        toSave.precision_t_NNDRT_80 = precision_t_NNDRT_80;
        toSave.recall_t_NNDRT_80 = recall_t_NNDRT_80;
        toSave.f1 = f1;
        toSave.f2 = f2;
        
        onlySaveParFor(namefile,toSave);
    end


    matchingScore = reshape(matchingScore,[],imgs_no/2);
    matchingScore_80 = reshape(matchingScore_80,[],imgs_no/2);
    legend0_str = reshape(legend0_str,[],imgs_no/2);
    legend1_str = reshape(legend1_str,[],imgs_no/2);
    legend2_str = reshape(legend2_str,[],imgs_no/2);
    repeatability_NN = reshape(repeatability_NN,[],imgs_no/2);
    precision_NN = reshape(precision_NN,[],imgs_no/2);
    recall_NN = reshape(recall_NN,[],imgs_no/2);
    precision_threshold = reshape(precision_threshold,[],imgs_no/2);
    recall_threshold = reshape(recall_threshold,[],imgs_no/2);
    precision_NNDRT = reshape(precision_NNDRT,[],imgs_no/2);
    recall_NNDRT = reshape(recall_NNDRT,[],imgs_no/2);
    precision_NNDRT_80 = reshape(precision_NNDRT_80,[],imgs_no/2);
    recall_NNDRT_80 = reshape(recall_NNDRT_80,[],imgs_no/2);
    res.score = matchingScore;
    res.score_80 = matchingScore_80;
    res.repeatability_NN = repeatability_NN;
    res.precision_NN = precision_NN;
    res.recall_NN = recall_NN;
    res.precision_threshold = precision_threshold;
    res.recall_threshold = recall_threshold;
    res.precision_NNDRT = precision_NNDRT;
    res.recall_NNDRT = recall_NNDRT;
    res.precision_NNDRT_80 = precision_NNDRT_80;
    res.recall_NNDRT_80 = recall_NNDRT_80;
    res.legend0 = legend0_str;
    res.legend1 = legend1_str;
    res.legend2 = legend2_str;
    
    fprintf('Program terminated normally.\n');

    % end everithing
    rng(s);
end

function [precision, recall] = getPrecisionRecallTheshold(twi, distanceM, legend)

    twi_vec = twi(:);
    num_of_corresp = sum(twi_vec);
    disp([legend '  number of correspondance is ' num2str(num_of_corresp)]);

    TP = 0;
    [~, idx] = sort(distanceM(:),'ascend');

    % %first half is all the points, then second half equally spaced
    % select = [round(1:size(precision_t_threshold,2)/(10000-1):size(precision_t_threshold,2)),size(precision_t_threshold,2)];
    % recall_t_threshold = recall_t_threshold(1,select);
    % precision_t_threshold = precision_t_threshold(1,select);

    % sizeIDX = min(size(distanceM,1),size(distanceM,2));
    sizeIDX = length(distanceM(:));
    recall = zeros(1,sizeIDX);
    precision = zeros(1,sizeIDX);

    if num_of_corresp ~= 0
        for id=1:sizeIDX % we are going through the order of increasing threshold
                         % see if current is right or wrong
            TP = TP + twi_vec(idx(id));
            FP = id - TP;
            recall(1,id) = TP/num_of_corresp;
            precision(1,id) = TP/(TP+FP);
        end
    end

    % Equally sample 10000 points
    select = [round(1:size(precision,2)/(10000-1):size(precision, ...
                                                      2)),size(precision,2)];

    precision = precision(1,select);
    recall = recall(1,select);


end

function [precision, recall] = getPrecisionRecall(twi0,twi1, distanceM, legend, sortingDirection)

    if ~exist('sortingDirection','var')
        sortingDirection = 'ascend';%'descend'
    end
    %%%%%%%%%%%%%%%%%%%%%%
    twi_vec = twi1(:);              % use this vector to
                                    % determine correctness

    num_of_corresp = sum(twi1(:));  % use this vector to count
                                    % correspondences

    disp([legend '  number of correspondance is ' num2str(num_of_corresp)]);

    TP = 0;
    [~, idx] = sort(distanceM(:),sortingDirection);
    sizeIDX = min(size(distanceM,1),size(distanceM,2));
    recall = zeros(1,sizeIDX);
    precision = zeros(1,sizeIDX);

    if num_of_corresp ~= 0
        for id=1:sizeIDX % we are going through the order of increasing threshold
                         % see if current is right or wrong
            TP = TP + twi_vec(idx(id));
            FP = id - TP;
            recall(1,id) = TP/num_of_corresp;
            precision(1,id) = TP/(TP+FP);
        end
    end
    %%%%%%%%%%%%%


end


function [list_parameters, list_parameters_name] = getParameters(path,list_method,parameters_dir_name)

    if ~exist('parameters_dir_name','var')
        parameters_dir_name = 'parameters';
    end

    nNumMethods = size(list_method, 1);
    list_parameters = cell(nNumMethods,1);
    list_parameters_name = cell(nNumMethods,1);
    %get the parameters
    for i_method=1:nNumMethods;%for each kp methods
        list_parameters{i_method,1} = [];%init with zero (default value)
        list_parameters_name{i_method,1} = 'none';%init with zero
                                                  %(default value)
        
        % Check if method dir exists (it might be blocked)
        if exist([path '/' list_method{i_method}], 'dir')
            method_folder = [path '/' list_method{i_method}];
        elseif exist([path '/_' list_method{i_method}], 'dir')
            method_folder = [path '/_' list_method{i_method}];
        else
            error(['Method folder does not exist. This should never ' ...
                   'happen']);
        end
        
        nfiles = dir([method_folder '/' parameters_dir_name '/']);
        nfiles = {nfiles.name};
        nfiles(ismember(nfiles,{'.','..'})) = [];
        idx = cellfun(@(x) strcmp(x(max(1,end-6):end),'.config'), nfiles,'UniformOutput',false);
        
        count = 1;
        for j=1:size(idx,2)
            if idx{j}
                %need to open the txt and load the parameters
                fileID = fopen([method_folder '/' parameters_dir_name '/' nfiles{j}]);
                try
                    list_parameters{i_method,count} = textscan(fileID,'%f');
                catch
                    list_parameters{i_method,count} = [];
                end
                name = nfiles{j};
                list_parameters_name{i_method,count} = name(1:end-7);
                fclose (fileID);
                %then save it in list_parametersKp{i}{count}
                count = count + 1;
            end
        end
        
    end
end

function [fileList di] = getAllFiles(dirName)

    dirData = dir(dirName);      %# Get the data for the current directory
    dirIndex = [dirData.isdir];  %# Find the index for directories
    fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
    if ~isempty(fileList)
        validIndex = ~ismember(fileList,{'.','..','.DS_Store'});
        fileList =  fileList(validIndex);
        fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                           fileList,'UniformOutput',false);
        
    end
    subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
    validIndex = ~ismember(subDirs,{'.','..','.DS_Store'});  %# Find index of subdirectories
                                                             %#   that are not '.' or '..'
    for iDir = find(validIndex)                  %# Loop over valid subdirectories
        nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
        fileList = [fileList; getAllFiles(nextDir)];  %# Recursively call getAllFiles
    end

    di = dirData(3:end);
    di(~dirIndex(3:end)) = '';
end

% function out = varname(var)
%   out = inputname(1);
% end

function [] = onlySaveParFor(namefile,toSave)

    match_score = toSave.match_score;
    match_score_80 = toSave.match_score_80;
    legend0 = toSave.legend0;
    legend1 = toSave.legend1;
    legend2 = toSave.legend2;
    repeatability_nn_cur = toSave.repeatability_nn_cur;
    precision_t_NN = toSave.precision_t_NN;
    recall_t_NN = toSave.recall_t_NN ;
    precision_t_threshold = toSave.precision_t_threshold;
    recall_t_threshold = toSave.recall_t_threshold;
    precision_t_NNDRT = toSave.precision_t_NNDRT;
    recall_t_NNDRT = toSave.recall_t_NNDRT;
    precision_t_NNDRT_80 = toSave.precision_t_NNDRT_80;
    recall_t_NNDRT_80 = toSave.recall_t_NNDRT_80;
    f1 = toSave.f1;
    f2 = toSave.f2;

    save( namefile ,'match_score','match_score_80','legend0','legend1','legend2','repeatability_nn_cur','precision_t_NN','recall_t_NN','precision_t_threshold','recall_t_threshold','precision_t_NNDRT','recall_t_NNDRT','precision_t_NNDRT_80','recall_t_NNDRT_80','f1','f2','-v7.3');
end


function D = distEmd( X, Y )

    Xcdf = cumsum(X,2);
    Ycdf = cumsum(Y,2);

    m = size(X,1);  n = size(Y,1);
    mOnes = ones(1,m); D = zeros(m,n);
    for i=1:n
        ycdf = Ycdf(i,:);
        ycdfRep = ycdf( mOnes, : );
        D(:,i) = sum(abs(Xcdf - ycdfRep),2);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% evaluateDescriptors.m ends here
