%  Copyright (c) 2014, Karen Simonyan
%  All rights reserved.
%  This code is made available under the terms of the BSD license (see COPYING file).

function Desc = get_desc(Patch, PR, varargin)

    Params = struct;
    Params.InitSigma = 1.4;
    Params.nAngleBins = 8;    
    Params.bNorm = true;
    Params.Proj = [];
    
    Params = vl_argparse(Params, varargin);
    
    % reshape
    if size(Patch, 2) == 1
        
        side = sqrt(numel(Patch));
        Patch = reshape(Patch, side, side);
    end
    
    % transform patch
    PatchTrans = trans_patch_T1(Patch, Params);

    % disp(size(PR));
    % disp(size(PatchTrans));
    
    % pool features
    Desc = (PR * PatchTrans)';
    
    % crop
    Desc = min(Desc, 1);
    
    % project
    if ~isempty(Params.Proj)
        Desc = Params.Proj * Desc(:);
    end
    
end