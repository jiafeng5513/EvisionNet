%%% saveFeatures.m --- 
%% 
%% Filename: saveFeatures.m
%% Description: 
%% Author: Yannick Verdie, Kwang Moo Yi
%% Maintainer: Kwang Moo Yi
%% Created: Wed Jun 29 10:04:55 2016 (-0700)
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


function [] = saveFeatures(feat,file)
    %for now only 6 param are acceptable because the 6th is a %d
%     if (size(feat,1) ~= 6)
%         error('for now only 6 parameters are acceptable because the 6th is a %d');
%     end
    
    nbVar = size(feat,1);

    fid = fopen(file, 'w');
    fprintf(fid, '%d\n',size(feat,1));%number of variables per kp
    fprintf(fid, '%d\n',size(feat,2));%number of kp
    
  
    formatVars = '%f %f %f %f %f %d ';
    if (nbVar > 6)
        formatVars = [formatVars repmat('%f ',1,nbVar-6)];
    end
    
    fprintf(fid, [formatVars '\n'], feat);
    fclose(fid);
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% saveFeatures.m ends here
