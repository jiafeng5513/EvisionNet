%%% originalEdgeFociDetector.m --- 
%% 
%% Filename: originalEdgeFociDetector.m
%% Description: 
%% Author: Kwang Moo Yi, Yannick Verdie
%% Maintainer: Kwang Moo Yi
%% Created: Wed Jun 29 13:43:59 2016 (-0700)
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


function [failed] = originalEdgeFociDetector(in_file_name, out_file_name, p, bMultipleOrientation)
    failed = false;
    
    global sRoot;    
    if (~exist('p','var'))
        p = struct();
    end
    rootFolder = sRoot; 

    if isfield(p,'rootFolder')
        rootFolder = p.rootFolder;
    end
    
    if (~exist('bMultipleOrientation','var'))
        bMultipleOrientation = false;
    end
    

    detector_path = [rootFolder '/external/src/EdgeFoci/EdgeFociAndBiCE.exe'];


    if (exist(detector_path) ~= 2)
        failed = true;
        return;
    end
    
    % Try using wine for linux and mac
    if ismac
        detector_path = ['/usr/local/bin/wine ' detector_path];
    elseif isunix
        detector_path = ['wine ' detector_path]; 
    end
    
    % Check if input image file is too large and halfsize it if needed --
    % the binaries fail otherwise
    bComputeHalfSize = false;
    img = imread(in_file_name);
    if size(img,1) > 2000
        bComputeHalfSize = true;
    end
    
    % Compute differently according to bComputeHalfSize
    if bComputeHalfSize == false
        % Do the original routine
        
        % Actual Computation using binary
        if bMultipleOrientation == false
            com = sprintf('%s -mi -co -i %s -o %s', detector_path, in_file_name, [out_file_name '.txt']);
        else
            com = sprintf('%s -mi -co -mo -i %s -o %s', detector_path, in_file_name, [out_file_name '.txt']);
        end
            
        system(com);
        [feat] = loadFeaturesEdgeFoci( [out_file_name '.txt']);
    else
        % Do the half-size routine
        
        % Save the half-size image
        img_half = imresize(img,0.5);
        in_file_half_name = strrep(in_file_name,'.png','_half.png');
        imwrite(img_half,in_file_half_name);        
        % Actual Computation using binary
        if bMultipleOrientation == false
            com = sprintf('%s -mi -co -i %s -o %s', detector_path, in_file_half_name, [out_file_name '_half.txt']);
        else
            com = sprintf('%s -mi -co -mo -i %s -o %s', detector_path, in_file_half_name, [out_file_name '_half.txt']);
        end
        system(com);
        [feat] = loadFeaturesEdgeFoci( [out_file_name '_half.txt']);    
        % REMEMBER TO RESCALE!
    end

    keypts(1:2,:) = feat(1:2,:);%x y
    
    keypts(4,:) = feat(7,:)*180/pi;%make it degree
    keypts(5,:) = feat(6,:);%score';
    keypts(6,:) = zeros(1,size(feat,2));
    
    % Beware Ss is not a,b,c 
    Ss = reshape([feat(3,:);feat(4,:);feat(4,:);feat(5,:)],2,2,[]);
    [m n t] = size(Ss);Sscell=squeeze(mat2cell(Ss, m, n, ones(1,t)));
    As = cellfun(@(x) mapFromS(mat2vec((x))),Sscell,'UniformOutput',false);
    As = cellfun(@(x) reshape(x,2,2),As,'UniformOutput',false);
    scale = cell2mat(cellfun(@(x) sqrt(det(x)),As,'UniformOutput',false));
    keypts(3,:) = scale';

    As2 = cellfun(@(x) inv(reshape(x,2,2)),As,'UniformOutput',false);
    S = cell2mat(cellfun(@(x) mat2vec(x'*x)',As2,'UniformOutput',false));

    keypts(7,:) = S(:,1);%a parameter of unoriented ellipse
    keypts(8,:) = S(:,2);%b parameter of unoriented ellipse
    keypts(9,:) = S(:,4);%c parameter of unoriented ellipse
    

    theta_new = keypts(4,:);
    theta_old = zeros(1,size(keypts,2));
    deltatheta = (theta_new - theta_old);
    R = arrayfun(@(x) [cosd(x) -sind(x); sind(x) cosd(x)],deltatheta, 'UniformOutput',false)';
    As = cellfun(@(Av,Rv)  mat2vec(Av*Rv),As,R ,'UniformOutput',false);


    keypts(10:13,:) =[As{:}];%save the affine at the end...
    
    % Rescale i.e. Enlarge by two!
    if bComputeHalfSize == true
        keypts = rescaleFeat(keypts,2.0);
    end
    
    saveFeatures(keypts,out_file_name);

end

% feat = [x,y,scale,deg,score,octave,a,b,c,Aff1,Aff2,Aff3,Aff4]
function [feat_out] = rescaleFeat(feat_in,rescale_by)

    % allocate mem
    feat_out = feat_in;
    
    % rescale x,y, and scale
    feat_out(1:3,:) = feat_in(1:3,:) * rescale_by;
    
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

% x y sxx sxy syy score orientation
function [feat_raw] = loadFeaturesEdgeFoci(file)
    fid = fopen(file, 'r');
    nb = fscanf(fid, '%d',1);
    feat_raw = fscanf(fid, '%f', [7, inf]);
    % end iv
    fclose(fid);
end

function vec = mat2vec( A )
    vec = A(:);
end

function [] = saveEdgeFociKeys(feat,file)
    fid = fopen(file, 'w');
    fprintf(fid,'%d\n',1);
    fprintf(fid,'%d\n',size(feat,1));
    for idxKey = 1:size(feat,1)
        fprintf(fid, '%f %f %f %f %f %f %f\n', feat(idxKey,1),feat(idxKey,2),feat(idxKey,7),feat(idxKey,8),feat(idxKey,9),feat(idxKey,5),feat(idxKey,4));
    end
    fclose(fid);    
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
%%% originalEdgeFociDetector.m ends here
