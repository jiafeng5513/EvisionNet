%%% getDescriptors_SURF.m --- 
%% 
%% Filename: getDescriptors_SURF.m
%% Description: 
%% Author: Kwang Moo Yi
%% Maintainer: 
%% Created: Thu Jun 16 17:51:42 2016 (+0200)
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


function [feat,desc,metric] = getDescriptors_SURF(img_info, kp_file_name, orientation_file_name, p)
    
    %here reject some kp
    listMethodsOK = {'SURF'};
    if (all(~strcmp(kp_file_name,listMethodsOK)))
        feat = [];
        desc = [];
        metric = '';
        return;
    end
    
    bUseCustomOrientation = ~isempty(strfind(orientation_file_name,'GHH'));

    metric = 'L2';
    methodName = strsplit(mfilename,'_');
    methodName = methodName{end};
    
    param_nameKp =  p.optionalParametersKpName;
    param_nameOrient =  p.optionalParametersOrientName;
    param_nameDesc =  p.optionalParametersDescName;
    
    paramname = [param_nameKp '-' param_nameOrient '-' param_nameDesc];
    
     in = img_info.image_name;
     in = strrep(in, 'image_gray', 'image_color');
%     
%     
%     forceRecomputeNoSave = false;
%     if isfield(p,'recomputeNoSave')
%         forceRecomputeNoSave = p.recomputeNoSave;
%     end
    
%     if forceRecomputeNoSave
%         kpf = [img_info.full_feature_prefix '_' kp_file_name '_keypoints-' param_nameKp '-txt'];
%         feat = [];
%         desc = [];
%     end
%     

    kpf = [img_info.full_feature_prefix '_' kp_file_name '_keypoints_' orientation_file_name '_oriented-' param_nameKp '-' param_nameOrient '-txt'];
    out = [img_info.full_feature_prefix '_' kp_file_name '_keypoints_' orientation_file_name '_oriented_' methodName '_descriptors-' paramname '-txt'];
    

    kpf2 = [img_info.full_feature_prefix '_' kp_file_name '_keypoints_for' methodName '_' orientation_file_name '_oriented-' param_nameKp '-' param_nameOrient '-txt'];
    if ~exist(out, 'file')% || forceRecomputeNoSave

        [feat_out, ~, ~] = loadFeatures(kpf);
        saveFeatures(feat_out,kpf2);

        warning('hack is running!');
        % To use custom orientations, we need to rescale a,b,c, and affine
        % so that we crop the region SURF descriptor is supposed to look at
        % and use the fix version
        kpf3 = [kpf2 '-rescale-for-fixed'];
        fRescale = (21.0 / 2.0 / 7.5) / (9.0 / 1.2 / 2.0);
        % fRescale = ((21.0*0.5) * 1.2 / 9.0 * 2.0) / 7.5;
        [feat_rescaled_for_surf] = rescaleEllipseAffine(feat_out,fRescale);
        saveFeatures(feat_rescaled_for_surf,kpf3);
        com = opencvFixedScaleDescriptor(methodName, in, kpf3, out,p);
        
        % % Re Scale it back
        % if (strcmp(kp_file_name,'SIFTFIXED')) %kpf -> /2 -> kpf2
        %     [feat_out, ~, ~] = loadFeatures(kpf2);
        %     [feat_out] = rescaleEllipseAffine(feat_out,coeffDescriptor/coeffTarget);
        %     saveFeatures(feat_out,kpf2);
        % end
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
%%% getDescriptors_SURF.m ends here
