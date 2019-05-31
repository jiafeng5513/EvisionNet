%%% termDisp.m --- 
%% 
%% Filename: termDisp.m
%% Description: 
%% Author: Kwang Moo Yi
%% Maintainer: 
%% Created: Tue Mar  8 19:35:15 2016 (+0100)
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

function [] = termDisp(ratio, prefix, suffix, dataset_name, num_key, scene_id)

    if strcmpi(dataset_name, 'strecha')
        dump_name = ['results/' prefix '/Strecha_fountain_' num2str(num_key) suffix '.mat'];
        load(dump_name);
        lg = result.legend(:,1);
        nn = result.AUC_NN(:);

        dump_name = ['results/' prefix '/Strecha_herzjesu_' num2str(num_key) suffix '.mat'];
        load(dump_name);
        nn = 0.5*nn + 0.5*result.AUC_NN(:);
        
    elseif strcmpi(dataset_name, 'dtu')
        idx_scene = 1;
        for i = scene_id
            
            dump_name = ['results/' prefix '/DTU_scene' num2str(i) '_' num2str(num_key) suffix '.mat'];

            load(dump_name);

            if idx_scene == 1
                lg = result.legend(:,1);
                nn = result.AUC_NN(:);
            else
                alpha = 1 / idx_scene;
                nn = (1-alpha) * nn + alpha * result.AUC_NN(:);
            end

            idx_scene = idx_scene + 1;
            
        end

    elseif strcmpi(dataset_name, 'webcam')

        seq_name = {'Chamonix', 'Courbevoie', 'Frankfurt', 'Mexico', 'Panorama', 'StLouis'};
        for idx_scene = 1:length(seq_name)
            
            dump_name = ['results/' prefix '/Webcam_' seq_name{idx_scene} '_' num2str(num_key) suffix '.mat'];

            load(dump_name);

            if idx_scene == 1
                lg = result.legend(:,1);
                nn = result.AUC_NN(:);
            else
                alpha = 1 / idx_scene;
                nn = (1-alpha) * nn + alpha * result.AUC_NN(:);
            end

            idx_scene = idx_scene + 1;
            
        end

    elseif strcmpi(dataset_name, 'viewpoints')

        seq_name = {'chatnoir', 'duckhunt', 'mario', 'outside', 'posters'};
        for idx_scene = 1:length(seq_name)
            
            dump_name = ['results/' prefix '/Viewpoints_' seq_name{idx_scene} '_' num2str(num_key) suffix '.mat'];

            load(dump_name);

            if idx_scene == 1
                lg = result.legend(:,1);
                nn = result.AUC_NN(:);
            else
                alpha = 1 / idx_scene;
                nn = (1-alpha) * nn + alpha * result.AUC_NN(:);
            end

            idx_scene = idx_scene + 1;
            
        end
        
    elseif strcmpi(dataset_name, 'edgefoci')

        seq_name = {'notredame', 'obama', 'paintedladies', 'rushmore', 'yosemite'};
        for idx_scene = 1:length(seq_name)
            
            dump_name = ['results/' prefix '/EdgeFoci_' seq_name{idx_scene} '_test_' num2str(num_key) suffix '.mat'];

            load(dump_name);

            if idx_scene == 1
                lg = result.legend(:,1);
                nn = result.AUC_NN(:);
            else
                alpha = 1 / idx_scene;
                nn = (1-alpha) * nn + alpha * result.AUC_NN(:);
            end

            idx_scene = idx_scene + 1;
            
        end

    else
        dump_name = ['results/' prefix '/' dataset_name '_' num2str(num_key) suffix '.mat'];
        load(dump_name);
        lg = result.legend(:,1);
        nn = result.AUC_NN(:);
        
    end


    display(' ');
    display('------------------------ ');
    display(' ');
    display('     AUC_NN      Legend');
    for i = 1:length(nn)
        if nn(i) > max(nn) * ratio
            display(sprintf('   %f    %s',nn(i),lg{i}));
        end
    end
    display('------------------------ ');
    display(' ');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% termDisp.m ends here
