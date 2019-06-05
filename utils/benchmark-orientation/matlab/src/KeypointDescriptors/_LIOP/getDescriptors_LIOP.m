%%% getDescriptors_LIOP.m --- 
%% 
%% Filename: getDescriptors_LIOP.m
%% Description: 
%% Author: Kwang Moo Yi, Yannick Verdie
%% Maintainer: Kwang Moo Yi, Yannick Verdie
%% Created: Wed Jun 29 09:22:07 2016 (-0700)
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


function [feat,desc,metric] = getDescriptors_LIOP(img_info, kp_file_name, orientation_file_name, p)
    
    %here reject some kp
    listMethodsOK = {'HARAFF','HESAFF'};
    if (all(~strcmp(kp_file_name,listMethodsOK)) || ~isempty(strfind(orientation_file_name,'GHH')))
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
   
    kpf = [img_info.full_feature_prefix '_' kp_file_name '_keypoints_' orientation_file_name '_oriented-' param_nameKp '-' param_nameOrient '-txt'];
    out = [img_info.full_feature_prefix '_' kp_file_name '_keypoints_' orientation_file_name '_oriented_' methodName '_descriptors-' paramname '-txt'];
    
 
    kpf2 = [img_info.full_feature_prefix '_' kp_file_name '_keypoints_for' methodName '_' orientation_file_name '_oriented-' param_nameKp '-' param_nameOrient '-txt'];
    if ~exist(out, 'file')
        
        [feat_out, ~, ~] = loadFeatures(kpf);
        saveFeatures(feat_out,kpf2);

        [feat, ~, ~] = loadFeatures(kpf2);
        
        if (size(feat,1) ~= 13 || any(sum(abs(feat(end-3:end,:))) == 0))
            error('Liop expects an affine transform for the patch as with Harris-affine for exemple...')
        end
 
%         [~,debug_img] = vl_covdet(single(img_info.image_gray), ...
%                                   'Frames',[feat(1:2,:);feat(end-3:end,:)],'descriptor','Patch');
%         debug_img = reshape(debug_img,[41,41]);
%         figure(1);imshow(imresize(uint8(debug_img),[200,200]));
%         drawnow;
        [f,desc] = vl_covdet(single(img_info.image_gray),'Frames',[feat(1:2,:);feat(end-3:end,:)],'descriptor','liop');
        
        if size(f,2) ~= size(feat,2)
           error('The LIOP descriptor deleted kp !'); 
        end

        
        saveDescriptors(desc,out);
      
    else
        if ~exist(kpf2, 'file')
            warning('for file does not exist, not using it');
        else
            kpf = kpf2;
        end    
        
        if ~exist(kpf, 'file')
            error('the keypoints do not exist, abort');
        end
        
        [feat, ~, ~] = loadFeatures(kpf);
        desc = loadDescriptors(out);
    end
    
     if (size(feat,2) ~= size(desc,2))
        error([methodName ' deleted kp, so now we have a missmatch !']);
    end
    
    if all(sum(feat(7:9,:) == 0))  %should never happen for LIOP because we use vl_feat
        %recpmpute, because cannot save it as runDescriptorsOpenCV is not
        %compatible (and the function delete kp, so the kp used are those from this function)
        feat(7,:) = 1./(feat(3,:).*feat(3,:));
        feat(8,:) =  zeros(size(feat(7,:),2),1)';
        feat(9,:) = feat(7,:);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_SIFT.m ends here

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% getDescriptors_LIOP.m ends here
