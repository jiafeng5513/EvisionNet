%%% loadFeatures.m --- 
%% 
%% Filename: loadFeatures.m
%% Description: 
%% Author: Yannick Verdie, Kwang Moo Yi
%% Maintainer: Kwang Moo Yi
%% Created: Wed Jun 29 10:03:48 2016 (-0700)
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


function [feat nbKp nbVar] = loadFeatures(file)
    fid = fopen(file, 'r');
    nbVar = fscanf(fid, '%d',1);
    nbKp = fscanf(fid, '%d',1);
    
    formatVars = '%f %f %f %f %f %d ';
    if (nbVar > 6)
        formatVars = [formatVars repmat('%f ',1,nbVar-6)];
    end
    feat = fscanf(fid, [formatVars '\n'], [nbVar, nbKp]); %, [nbVar, nbKp]);

    % compute a b c if we don't have it
    if (size(feat,1) < 9 && size(feat,2) > 0) % only if we have at least one kp
        feat(7,:) = 1./(feat(3,:).*feat(3,:));
        feat(8,:) =  zeros(size(feat(7,:),2),1)';
        feat(9,:) = feat(7,:);        
    end
    
    % compute Affine if we don't have it
    if (size(feat,1) < 13 && size(feat,2) > 0) % only if we have at least one kp

        % Conversion code from EdgeFoci
        Ss = reshape([feat(7,:);feat(8,:);feat(8,:);feat(9,:)],2,2,[]);
        [m n t] = size(Ss);Sscell=squeeze(mat2cell(Ss, m, n, ones(1,t)));
        As = cellfun(@(x) mapFromS(mat2vec(inv(x))),Sscell,'UniformOutput',false);
        As = cellfun(@(x) reshape(x,2,2),As,'UniformOutput',false);
        
        % Rotate accordingly
        theta_new = feat(4,:);
        theta_old = zeros(1,size(feat,2));
        deltatheta = (theta_new - theta_old);
        R = arrayfun(@(x) [cosd(x) -sind(x); sind(x) cosd(x)],deltatheta, 'UniformOutput',false)';
        As = cellfun(@(Av,Rv)  mat2vec(Av*Rv),As,R ,'UniformOutput',false);
        
        feat(10:13,:) =[As{:}];
    end

    fclose(fid);
end

function vec = mat2vec( A )
    vec = A(:);
end


function A = mapFromS(S)
    % --------------------------------------------------------------------
    % Returns the (stacking of the) 2x2 matrix A that maps the unit circle
    % into the ellipses satisfying the equation x' inv(S) x = 1. Here S
    % is a stacked covariance matrix, with elements S11, S12 and S22.
    %
    % The goal is to find A such that AA' = S. In order to let the Y
    % direction unaffected (upright feature), the assumption is taht
    % A = [a b ; 0 c]. Hence
    %
    %  AA' = [a^2, ab ; ab, b^2+c^2] = S. %nop, to me it is
    %AA' = [a^2+b^2, bc ; bc, c^2] = S

%     A = zeros(4,size(S,2)) ;
%     c = sqrt(S(3,:));
%     b = S(2,:) ./ max(c, 1e-18) ;
%     A(1,:) = sqrt(max(S(1,:) - b.*b, 0)) ;
%     A(2,:) = b ;
%     A(4,:) = c ;
    a = sqrt(S(1,:));
    b = S(2,:) ./ max(a, 1e-18) ;

    A(1,:) = a ;
    A(2,:) = b ;
    A(4,:) = sqrt(max(S(4,:) - b.*b, 0)) ;
end






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% loadFeatures.m ends here
