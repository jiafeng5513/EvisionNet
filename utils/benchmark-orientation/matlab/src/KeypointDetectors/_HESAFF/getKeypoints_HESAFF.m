%% getKeypoints_HESAFF.m --- 
% 
% Filename: getKeypoints_HESAFF.m
% Description: Wrapper Function for HESAFF
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:20:54 2015 (+0200)
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


function [keypts] = getKeypoints_HESAFF(img_info, p)
    methodName = strsplit(mfilename,'_');
    methodName = methodName{end};
    
    param_nameKp =  p.optionalParametersKpName;
    %param_nameOrient =  p.optionalParametersOrientName;
    %param_nameDesc =  p.optionalParametersDescName;

    out = [img_info.full_feature_prefix '_' methodName '_keypoints-' param_nameKp '-txt'];    
    %sift_name = [img_info.full_feature_prefix '_' methodName '_keypoints-' param_nameKp '.mat'];
    if ~exist(out, 'file') 
        in = img_info.image_gray;
        out = [img_info.full_feature_prefix '_' methodName '_keypoints-' param_nameKp '-txt'];
        
        [feat,~,INFO] = vl_covdet(single(in),'method','HessianLaplace','DoubleImage',false,'EstimateOrientation', true, 'EstimateAffineShape', true, 'PeakThreshold', 100);
        
        keypts = [feat(1:2,:); zeros(5,size(feat,2))];
        A = reshape(feat(3:6,:),2,2,[]) ;
        
        dets = zeros(1,size(A,3));
        for i=1:size(A,3)
            As = squeeze(A(:,:,i));
            dets(i) = det(As);
            
            b = [0;1;0];
            As(3,3) = 0;%make it 3x3 with zeros padding
            a = As *b;
            a = a/norm(a);
            c = cross(a,b);
            
            keypts(4,i) = mod(-1*atan2d(c(3),dot(a,b)),360);%the clockwise is oposite, so *-1
%              clf
%              vl_plotframe(feat(:,i))
%              vl_plotframe([feat(1:2,i);sqrt(dets(i));keypts(4,i)/180*pi])
            As = inv(As(1:2,1:2));
%             As = (As(1:2,1:2));
            S = As'*As;
            keypts(7,i) = S(1,1);%a parameter of unoriented ellipse
            keypts(8,i) = S(1,2);%b parameter of unoriented ellipse
            keypts(9,i) = S(2,2);%c parameter of unoriented ellipse
%             drawellipse([keypts(7,i) keypts(8,i); keypts(8,i) keypts(9,i) ], keypts(1,i),keypts(2,i),'y');
        end
        
        scale = sqrt(dets);
        score  = INFO.peakScores; 
        octave = zeros(1,size(feat,2)); 

        keypts(5,:) = score';
        keypts(3,:) = scale';
        keypts(6,:) = octave';
        
        keypts(10:13,:) = feat(3:6,:);%save the affine at the end...
        
        
        %now sort the kp Harris by score
        [~,idx] = sort(score,'descend');
        keypts = keypts(:,idx);
        
        
        saveFeatures(keypts,out);
        
        %save(sift_name, 'keypts', '-v7.3');
    else
        [keypts, ~, ~] = loadFeatures(out);
        keypts = keypts';
        %loadkey = load(sift_name);
        %keypts = loadkey.keypts;
    end
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

    A = zeros(4,size(S,2)) ;
    c = sqrt(S(3,:));
    b = S(2,:) ./ max(c, 1e-18) ;
    A(1,:) = sqrt(max(S(1,:) - b.*b, 0)) ;
    A(2,:) = b ;
    A(4,:) = c ;
%     a = sqrt(S(1,:));
%     b = S(2,:) ./ max(a, 1e-18) ;
% 
%     A(1,:) = a ;
%     A(2,:) = b ;
%     A(4,:) = sqrt(max(S(3,:) - b.*b, 0)) ;
end

function drawellipse(Mi,i,j,col)
hold on;
[v e]=eig(Mi);

l1=1/sqrt(e(1));

l2=1/sqrt(e(4));

alpha=atan2(v(4),v(3));
s=1;
t = 0:pi/50:2*pi;
y=s*(l2*sin(t));
x=s*(l1*cos(t));

xbar=x*cos(alpha) + y*sin(alpha);
ybar=y*cos(alpha) - x*sin(alpha);
plot(ybar+i,xbar+j,'-k','LineWidth',3);
plot(ybar+i,xbar+j,col,'LineWidth',1);
col='-k';
%plot([i-2 i+2],[j j],col,'LineWidth',3);
%plot([i i],[j-2 j+2],col,'LineWidth',3);
set(gca,'Position',[0 0 1 1]);
hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_HESAFF.m ends here
