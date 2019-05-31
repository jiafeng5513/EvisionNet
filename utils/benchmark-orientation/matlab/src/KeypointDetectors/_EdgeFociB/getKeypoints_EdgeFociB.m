%% getKeypoints_EdgeFociB.m --- 
% 
% Filename: getKeypoints_EdgeFociB.m
% Description: Wrapper Function for EdgeFoci to be used with BiCE
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

function [keypts] = getKeypoints_EdgeFociB(img_info, p)
    methodName = strsplit(mfilename,'_');
    methodName = methodName{end};
    
    param_nameKp =  p.optionalParametersKpName;
    out = [img_info.full_feature_prefix '_' methodName '_keypoints-' param_nameKp '-txt'];
    if ~exist(out, 'file')

        in = img_info.image_name;
        in = strrep(in, 'image_gray', 'image_color');
        originalEdgeFociDetector( in, out, p);
    end
   
    [keypts, ~, ~] = loadFeatures(out);
    keypts = keypts';

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_EdgeFociB.m ends here
