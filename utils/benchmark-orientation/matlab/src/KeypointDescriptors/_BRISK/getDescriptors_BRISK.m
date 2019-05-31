%%% getDescriptors_BRISK.m --- 
%% 
%% Filename: getDescriptors_BRISK.m
%% Description: 
%% Author: Kwang Moo Yi
%% Maintainer: 
%% Created: Thu Jun 16 17:48:46 2016 (+0200)
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

function [feat,desc,metric] = getDescriptors_BRISK(img_info, kp_file_name, orientation_file_name, p)
    %here reject some kp
    listMethodsOK = {'BRISKB'};
    if (all(~strcmp(kp_file_name,listMethodsOK)) || ~isempty(strfind(orientation_file_name,'GHH')))
        feat = [];
        desc = [];
        metric = '';
        return;
    end

    metric = 'Hamming';
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
        
        com = opencvDescriptor(methodName, in, kpf2, out,p);
    end
    kpf = kpf2;
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

% feat = [x,y,scale,deg,score,octave,a,b,c,Aff1,Aff2,Aff3,Aff4]
function [feat_out] = rescaleEllipseAffine(feat_in,rescale_by)

    % allocate mem
    feat_out = feat_in;


    for i=1:size(feat_in,2)%%dim 2 to check
        
        %%apply to scale
        %static const float basicSize06=basicSize_*0.6;
        %basicscale=std::max((int)(scales_/lb_scalerange*(log(1.45*basicSize_/(basicSize06))/log2)+0.5),0);
        
        % rescale Affine
        feat_out(10:13,:) = feat_in(10:13,:) * rescale_by;

        % recompute a,b,c from affine
        A = reshape(feat_out(10:13,:),2,2,[]) ;        
        dets = zeros(1,size(A,3));

        
        As = squeeze(A(:,:,i));
        As2 = inv(As(1:2,1:2));
        %             As = (As(1:2,1:2));
        S = As2'*As2;
        feat_out(7,i) = S(1,1);%a parameter of unoriented ellipse
        feat_out(8,i) = S(1,2);%b parameter of unoriented ellipse
        feat_out(9,i) = S(2,2);%c parameter of unoriented ellipse
        
        %%to remember...
        %sqrt(det(As)) == sqrt(det(((chol(S))))) == sqrt(sqrt(det(inv(S))))
    end
        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% getDescriptors_BRISK.m ends here
