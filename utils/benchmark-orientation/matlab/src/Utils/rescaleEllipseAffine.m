%%% rescaleEllipseAffine.m --- 
%% 
%% Filename: rescaleEllipseAffine.m
%% Description: 
%% Author: Yannick
%% Maintainer: 
%% Created: Fri Oct 23 18:49:35 2015 (+0200)
%% Version: 
%% Package-Requires: ()
%% Last-Updated: Fri Oct 30 16:16:29 2015 (+0100)
%%           By: Kwang
%%     Update #: 2
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
%% 
%% Copyright (C), EPFL Computer Vision Lab.
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%%% Code:

function [feat_out] = rescaleEllipseAffine(feat_in,rescale_by)

% allocate mem
    feat_out = feat_in;
    
    feat_out(3,:) = feat_in(3,:) * rescale_by;

    % rescale Affine
    feat_out(10:13,:) = feat_in(10:13,:) * rescale_by;
    
    % recompute a,b,c from affine
    A = reshape(feat_out(10:13,:),2,2,[]) ;        
    dets = zeros(1,size(A,3));
    for i=1:size(A,3)
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
%%% rescaleEllipseAffine.m ends here
