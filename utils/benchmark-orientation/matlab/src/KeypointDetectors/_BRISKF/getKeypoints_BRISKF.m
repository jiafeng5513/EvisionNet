%% getDescriptors_BRISKF.m --- 
% 
% Filename: getKeypoints_BRISKF.m
% Description: Wrapper Function for BRISK to be used with FREAK
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:20:35 2015 (+0200)
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


function [keypts] = getKeypoints_BRISKF(img_info, p)
     methodName = strsplit(mfilename,'_');
    methodName = methodName{end};
    
    
    param_nameKp =  p.optionalParametersKpName;
    out = [img_info.full_feature_prefix '_' methodName '_keypoints-' param_nameKp '-txt'];
    if ~exist(out, 'file')

        in = img_info.image_name;
        in = strrep(in, 'image_gray', 'image_color');
        
        %remove the F
        originalDetector(methodName(1:end-1), in, out)
    
    end
   
    [keypts, ~, ~] = loadFeatures(out);
    keypts = keypts';

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_BRISKF.m ends here
