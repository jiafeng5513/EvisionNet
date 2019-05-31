%  Copyright (c) 2014, Karen Simonyan
%  All rights reserved.
%  This code is made available under the terms of the BSD license (see COPYING file).

function PatchTrans = trans_patch_T1(Patch, Params)

    Patch = single(Patch);
    
    % smooth
    if ~isempty(Params.InitSigma)
        Patch = vl_imsmooth(Patch, Params.InitSigma);
    end
    
    % compute gradient
    GradFilt = [-1 0 1];
    
    Ix = imfilter(Patch, GradFilt, 'replicate');
    Iy = imfilter(Patch, GradFilt', 'replicate');
    
    % gradient magnitude
    GMag = sqrt(Ix .^ 2 + Iy .^ 2);
    
    % gradient orientation: [0; 2 * pi]
    GAngle = atan2(Iy, Ix) + pi;
    
    % soft-assignment of gradients to the orientation histogram
    AngleStep = 2 * pi / Params.nAngleBins;
    
    GAngleRatio = GAngle / AngleStep - 0.5;
    Offset1 = mod(GAngleRatio, 1);
    w1 = 1 - Offset1;
    w2 = Offset1;
    
    Bin1 = ceil(GAngleRatio);
    Bin1(Bin1 == 0) = Params.nAngleBins;
    
    Bin2 = Bin1 + 1;
    Bin2(Bin2 > Params.nAngleBins) = 1;
    
    % magnitude normalisation
    if Params.bNorm
        
        Quantile = 0.8;
            
        T = quantile(GMag(:), Quantile);
        
        T = T / Params.nAngleBins;
        
        % normalise magnitude
        GMag = GMag / T;
    end

    % feature channels
    PatchTrans = zeros(numel(Patch), Params.nAngleBins, 'single');
    
    for iBin = 1:Params.nAngleBins
        
        GMagCur = zeros(size(Patch));
        
        Mask1 = (Bin1 == iBin);
        GMagCur(Mask1) = w1(Mask1) .* GMag(Mask1);
        
        Mask2 = (Bin2 == iBin);
        GMagCur(Mask2) = w2(Mask2) .* GMag(Mask2);
        
        PatchTrans(:, iBin) = GMagCur(:);         
        
    end   
        
end