%%% getDescriptors_DAISY.m --- 
%% 
%% Filename: getDescriptors_DAISY.m
%% Description: 
%% Author: Kwang Moo Yi, Yannick Verdie
%% Maintainer: Kwang Moo Yi, Yannick Verdie
%% Created: Wed Jun 29 09:15:37 2016 (-0700)
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


function [feat,desc,metric] = getDescriptors_DAISY(img_info, kp_file_name, orientation_file_name, p)
    %here reject some kp
    listMethodsOK = {'HARAFF','EdgeFociD','EdgeFociDMult'};
    if (all(~strcmp(kp_file_name,listMethodsOK)))
        feat = [];
        desc = [];
        metric = '';
        return;
    end

    metric = 'L2'; %I am not sure
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
    if ~exist(out, 'file')% || forceRecomputeNoSave

        [feat_out, ~, ~] = loadFeatures(kpf);
        saveFeatures(feat_out,kpf2);
        %originalDAISYDescriptor( in, kpf, out,p);
        com = originalDescriptor(methodName, in, kpf2, out,p);
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
    
    if all(sum(feat(7:9,:) == 0))
        %recpmpute, because cannot save it as runDescriptorsOpenCV is not
        %compatible (and the function delete kp, so the kp used are those from this function)
        feat(7,:) = 1./(feat(3,:).*feat(3,:));
        feat(8,:) =  zeros(size(feat(7,:),2),1)';
        feat(9,:) = feat(7,:);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% getDescriptors_DAISY.m ends here
