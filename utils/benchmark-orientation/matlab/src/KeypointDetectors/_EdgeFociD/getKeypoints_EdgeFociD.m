%% getKeypoints_EdgeFociD.m --- 
% 
% Filename: getKeypoints_EdgeFociD.m
% Description: Wrapper Function for EdgeFoci with Daisy
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:16:41 2015 (+0200)
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
%% Code:

function [keypts] = getKeypoints_EdgeFociD(img_info, p)
    methodName = strsplit(mfilename,'_');
    methodName = methodName{end};
    
    param_nameKp =  p.optionalParametersKpName;
    out = [img_info.full_feature_prefix '_' methodName '_keypoints-' param_nameKp '-txt'];
    if ~exist(out, 'file')

        in = img_info.image_name;
        in = strrep(in, 'image_gray', 'image_color');
        originalEdgeFociDetector( in, out, p, false);
    end
   
    [keypts, ~, ~] = loadFeatures(out);
    keypts = keypts';

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_EdgeFociD.m ends here
