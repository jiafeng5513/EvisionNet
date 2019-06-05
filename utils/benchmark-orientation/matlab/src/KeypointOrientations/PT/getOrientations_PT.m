%% getKeypoints_SIFT.m --- 
% 
% Filename: getKeypoints_SIFT.m
% Description: Wrapper Function for SIFT
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:20:35 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:20:49 2015 (+0200)
%           By: Kwang
%     Update #: 1
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
%% Code:


function [feat] = getOrientations_PT(img_info, kp_file_name, p)

    param_nameKp =  p.optionalParametersKpName;
    param_nameOrient =  p.optionalParametersOrientName;

    out = [img_info.full_feature_prefix '_' kp_file_name '_keypoints_PT_oriented-' param_nameKp '-' param_nameOrient '-txt'];

    if ~exist(out, 'file')

        %but for None, do nothing
        % save 
        in = [img_info.full_feature_prefix '_' kp_file_name '_keypoints-' param_nameKp '-txt'];
%         [feat, ~, ~] = loadFeatures(in);
%         saveFeatures(feat,out);
         copyfile(in, out);
        
    end
    [feat, ~, ~] = loadFeatures(out);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_SIFT.m ends here
