%% computeDescriptors.m ---
%
% Filename: computeDescriptors.m
% Description:
% Author: Yannick Verdie, Kwang Moo Yi
% Maintainer: Yannick Verdie, Kwang Moo Yi
% Created: Tue Jun 16 17:13:09 2015 (+0200)
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
% Copyright (C), EPFL Computer Vision Lab.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%% Code:


function Allrepeatability = computeDescriptors(parameters)

    models = parameters.models;
    testsets = parameters.testsets;
    numberOfKeypoints  = parameters.numberOfKeypoints;

    % Default display to Matching Scores
    whatToDisplay = {'MatchingScore'};    
    if isfield(parameters,'whatToDisplay')
        whatToDisplay = parameters.whatToDisplay;
    end
    
    % for individual precision recall graphs
    targetMatchingLegend = '';
    if (isfield(parameters,'optionalMatchingLegend'))
        targetMatchingLegend = parameters.optionalMatchingLegend;
    end
    
    
    combinations = cell(length(models)*length(testsets),2);
    idxComb = 1;
    for idxModel = 1:length(models)
        for idxTestSet = 1:length(testsets)
            combinations{idxComb,1} = models{idxModel};
            combinations{idxComb,2} = testsets{idxTestSet};
            combinations{idxComb,3} = numberOfKeypoints{idxTestSet};
            idxComb = idxComb + 1;
        end
    end

    Allrepeatability = cell(size(combinations,1),1);
    for idxComb = 1:size(combinations,1)
        
        % Check if somebody else is running
        lockfile = ['.lock/' combinations{idxComb,1} '-' ...
                    strrep(combinations{idxComb,2},'/','-') '-' ...
                    num2str(combinations{idxComb,3}) '.lock'];
        if exist(lockfile,'file')
            continue;
        end
        % Create a lock
        lockfid = fopen(lockfile,'w');
        fprintf(lockfid,'Working on it\n');
        fclose(lockfid);
        
        
        res = evaluateDescriptors(combinations{idxComb,1}, ...
                                  combinations{idxComb,2}, ...
                                  combinations{idxComb,3}, ...
                                  parameters);
        disp(['Trained with ' combinations{idxComb,1} ...
              ' and tested on ' combinations{idxComb,2}]);               
        
        possibleDisplay = {'BarGraph', 'MatchingScore','MatchingScore_80', ...
                           'AUC_PrecisionRecall_NN', ...
                           'AUC_PrecisionRecall_threshold', ...
                           'AUC_PrecisionRecall_NNDRT', ...
                           'individual_PrecisionRecall_NN', ...
                           'individual_PrecisionRecall_threshold', ...
                           'individual_PrecisionRecall_NNDRT', ...
                           'AUC_PrecisionRecall_NNDRT_80', ...
                           'Repetability'}
        whatToDisplay = ismember(possibleDisplay,parameters.whatToDisplay);

        if (whatToDisplay(1))
            displayBarGraph(combinations,idxComb,res);
        end

        AVG_MatchingScore = [];
        if (whatToDisplay(2))
            AVG_MatchingScore = displayAVGBarGraph(combinations, ...
                                                   idxComb, ...
                                                   res.score, res);
        end
        
        if (whatToDisplay(3))
            AVG_MatchingScore_80 = displayAVGBarGraph(combinations, ...
                                                      idxComb, ...
                                                      res.score_80, ...
                                                      res, '(80%)');
        end
        
        AUC_NN = [];
        if (whatToDisplay(4))
            AUC_NN = displayAVG_AUC_PrecisionRecall(combinations,idxComb, ...
                                                    res.precision_NN, ...
                                                    res.recall_NN,res.legend0, ...
                                                    '(NN) ');
        end
        
        AUC_TH = [];
        if (whatToDisplay(5))
            AUC_TH = displayAVG_AUC_PrecisionRecall(combinations, ...
                                                    idxComb, ...
                                                    res.precision_threshold, ...
                                                    res.recall_threshold, ...
                                                    res.legend0, '(Threshold) ');
        end
        
        AUC_NNDRT = [];
        if (whatToDisplay(6))
            AUC_NNDRT = displayAVG_AUC_PrecisionRecall(combinations, ... 
                                                       idxComb, ...
                                                       res.precision_NNDRT, ...
                                                       res.recall_NNDRT, ...
                                                       res.legend0, '(NNDRT) ');
        end
        
        if (whatToDisplay(10))
            displayPrecisionRecall(combinations, ...
                                   idxComb, ...
                                   res.precision_NNDRT_80, ...
                                   res.recall_NNDRT_80, ...
                                   res.legend1, ...
                                   '(NNDRT 80%) ', ...
                                   targetMatchingLegend );
        end  
        
        if (whatToDisplay(7))
            displayPrecisionRecall(combinations, ...
                                   idxComb, ...
                                   res.precision_NN, ...
                                   res.recall_NN, ...
                                   res.legend1, ...
                                   '(NN) ', ...
                                   targetMatchingLegend );
        end      
        
        if (whatToDisplay(8))
            displayPrecisionRecall(combinations, ...
                                   idxComb, ... 
                                   res.precision_threshold, ...
                                   res.recall_threshold, ...
                                   res.legend1, ...
                                   '(Threshold) ', ...
                                   targetMatchingLegend );
        end    
        
        if (whatToDisplay(9))
            displayPrecisionRecall(combinations,idxComb,res.precision_NNDRT, ...
                                   res.recall_NNDRT,res.legend1, '(NNDRT) ', ...
                                   targetMatchingLegend );
        end    
        
        if (whatToDisplay(11))
            displayRepetability(combinations,idxComb,res);
        end
        
        result.AVG_MatchingScore = AVG_MatchingScore;
        result.AUC_NN = AUC_NN;
        result.AUC_NNDRT = AUC_NNDRT;
        result.AUC_TH = AUC_TH;
        result.legend = res.legend0;

        if ~exist('results','dir')
            mkdir('results');
        end
        if ~exist('results_BIG','dir')
            mkdir('results_BIG');
        end
        if ~exist(['results/', parameters.models{1}],'dir')
            mkdir(['results/', parameters.models{1}]);
        end
        if ~exist(['results_BIG/',parameters.models{1}] ,'dir')
            mkdir(['results_BIG/', parameters.models{1}]);
        end
        save(['results/' parameters.models{1} '/' ...
              strrep(combinations{idxComb,2},'/','_') '_' ...
              num2str(numberOfKeypoints{1})], 'result', '-v7.3');
        save(['results_BIG/' parameters.models{1} '/' ...
              strrep(combinations{idxComb,2},'/','_') '_' ...
              num2str(numberOfKeypoints{1})], 'res', '-v7.3');
        
        % Release the lock
        delete(lockfile);
    end

end

function [] = displayRepetability(combinations,idxComb,res)
    reps = mean(res.repeatability_NN,2)';
    
    for iconfig=1:size(reps,2)
        disp(['Average repeatability for Config:' num2str(iconfig) ...
              ' ' res.legend0{iconfig,1}  ': ' num2str(reps(iconfig))])
    end
    disp(['===>Average Repetability: ' num2str(mean(reps)) ' <===']);
    fprintf('\n');

    
    if (feature('ShowFigureWindows'))
        %bar graph
        colors = hsv(size(reps,2));
        figure;
        for i=1:size(reps,2)
            bar(i,reps(1,i), 'facecolor', colors(i,:));
            hold on;
        end
        hold off;
        
        title([optionText 'Repeatability: trained with ' ...
               combinations{idxComb,1} ' and tested on ' ...
               combinations{idxComb,2} ])
        set(gca, 'XTick', 1:size(reps,2))
        legend(res.legend0{:,1}, 'Location','SouthEast');
        text(1:size(reps,2),reps(1,:)',num2str(reps(1,:)','%0.2f'), ...
             'HorizontalAlignment','center','VerticalAlignment','bottom')
        
        drawnow;
    end

end

function [] = displayBarGraph(combinations,idxComb,res)
    if (feature('ShowFigureWindows'))
        for iconfig=1:size(res.score,1)
            %=========================================for display
            colors = hsv(size(res.score,2));
            figure;
            for i=1:size(res.score,2)
                bar(i,res.score(iconfig,i), 'facecolor', colors(i,:));
                hold on;
            end
            hold off;
            title(['Trained with ' combinations{idxComb,1} ' and tested on ' combinations{idxComb,2} '. Config:' num2str(iconfig) ' (' res.configName{iconfig,1} ')'])
            set(gca, 'XTick', 1:size(res.score,2))
            legend(res.legend1{iconfig,:}, 'Location','SouthEast');
            text(1:size(res.score,2),res.score(iconfig,:)',num2str(res.score(iconfig,:)','%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom')
            drawnow;
        end
    end
end

function [avg] = displayAVGBarGraph(combinations,idxComb,score,res, optionalLegend)
    
    if ~exist('optionalLegend','var')
        optionalLegend = '';
    end

    avg = mean(score,2)';
    for iconfig=1:size(avg,2)
        disp(['Average matching score ' optionalLegend ' for Config:' num2str(iconfig) ' ' res.legend0{iconfig,1}  ': ' num2str(avg(iconfig))])
    end
    disp(['===>Average matching score ' optionalLegend ': ' num2str(mean(avg)) ' <===']);        fprintf('\n');


    if (feature('ShowFigureWindows'))


        %=========================================for display
        colors = hsv(size(avg,2));
        figure;
        for i=1:size(avg,2)
            bar(i,avg(i), 'facecolor', colors(i,:));
            hold on;
        end
        hold off;
        title(['Mathcing Score ' optionalLegend ' | ' combinations{idxComb,1} ' / ' combinations{idxComb,2} ' | ' num2str(iconfig) ' (' res.legend2{iconfig,1} ')'])
        set(gca, 'XTick', 1:size(score,2))
        legend(res.legend0{:,1}, 'Location','SouthEast');
        text(1:size(avg,2),avg(:)',num2str(avg(:),'%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom')
        drawnow;
    end
end

function [val_1nn2nn] = display1NN2NN(combinations,idxComb,res)

    val_1nn2nn= zeros(1,size(res.score,1));

    for iconfig=1:size(res.score,1)
        val = mean(cell2mat({res.distanceScore{iconfig,:}}'),1);
        val_1nn2nn(1,iconfig) = mean(val);
        disp(['1NN/2NN for Config:' num2str(iconfig) ' (' res.configName{iconfig,1} '): ' num2str(val_1nn2nn(1,iconfig))])
    end
    disp(['===>Average 1NN/2NN: ' num2str(mean(val_1nn2nn)) ' <===']);        fprintf('\n');


    if (feature('ShowFigureWindows'))
        for iconfig=1:size(res.score,1)
            val = mean(cell2mat({res.distanceScore{iconfig,:}}'),1);
            figure;
            plot(1:size(val,2),val);
            title(['1NN/2NN: Trained with ' combinations{idxComb,1} ' and tested on ' combinations{idxComb,2} '. Config:' num2str(iconfig) ' (' res.configName{iconfig,1} ')'])
            set(gca, 'XTick', 1:size(val,2))
            text(1:size(val,2),val',num2str(val','%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom')
            axis([0 1 0 1]);
            drawnow;
        end
    end
end

function [AUC] = getavgAUCPrecisionRecall(prec,rec)

    AUC = zeros(1,size(prec,1));

    for iconfig=1:size(prec,1)
        valsum = zeros(1,size(prec,2));
        for ipair=1:size(prec,2)
            valX = prec{iconfig,ipair};
            valX = 1 - valX;
            valY = rec{iconfig,ipair};
            
            % two at the beginning to open
            valY = [0,valY]; valX = [valX(1),valX];
            valY = [0,valY]; valX = [0,valX];
            
            % two at the end to close
            valY(end+1) = valY(end); valX(end+1)=1;
            valY(end+1) = 1; valX(end+1)=1;
            
            % valsum(1,ipair) = trapz(valX,valY);
            valsum(1,ipair) = trapz(valY,1-valX);
        end
        AUC(1,iconfig) = mean(valsum);
    end
end


function [] = subplot(t,data)
% - Define dummy data: 11 time series.
%      t       = 0 : 0.1 : 10 ;
%      data    = 2 * repmat( sin(t).', 1,11 ) + rand( length(t), 11 ) ;
    nSeries = size( data, 2 ) ;
    % - Build figure.
    figure() ;  clf ;
    set( gcf, 'Color', 'White', 'Unit', 'Normalized', ...
              'Position', [0.1,0.1,0.6,0.6] ) ;
    % - Compute #rows/cols, dimensions, and positions of lower-left corners.
    nCol = 4 ;  nRow = ceil( nSeries / nCol ) ;
    rowH = 0.58 / nRow ;  colW = 0.7 / nCol ;
    colX = 0.06 + linspace( 0, 0.96, nCol+1 ) ;  colX = colX(1:end-1) ;
    rowY = 0.1 + linspace( 0.9, 0, nRow+1 ) ;  rowY = rowY(2:end) ;
    % - Build subplots axes and plot data.
    for dId = 1 : nSeries
        rowId = ceil( dId / nCol ) ;
        colId = dId - (rowId - 1) * nCol ;
        axes( 'Position', [colX(colId), rowY(rowId), colW, rowH] ) ;
        plot( t, data(:,dId), 'b' ) ;
        grid on ;
        xlabel( '\theta(t) [rad]' ) ;  ylabel( 'Anomaly [m]' ) ;
        title( sprintf( 'Time series %d', dId )) ;    
    end
    % - Build title axes and title.
    axes( 'Position', [0, 0.95, 1, 0.05] ) ;
    set( gca, 'Color', 'None', 'XColor', 'White', 'YColor', 'White' ) ;
    text( 0.5, 0, 'My Nice Title', 'FontSize', 14', 'FontWeight', 'Bold', ...
          'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
end

function [AUC] = displayAVG_AUC_PrecisionRecall(combinations,idxComb, precision,recall,legend0, optionText)

    AUC = getavgAUCPrecisionRecall(precision,recall);

    for iconfig=1:size(AUC,2)
        disp(['Average AUC precision-recall for Config:' num2str(iconfig) ' ' legend0{iconfig,1}  ': ' num2str(AUC(iconfig))])
    end
    disp([optionText '===>Average AUC: ' num2str(mean(AUC)) ' <===']);        fprintf('\n');


    if (feature('ShowFigureWindows'))
        %bar graph
        colors = hsv(size(AUC,2));
        figure;
        for i=1:size(AUC,2)
            bar(i,AUC(1,i), 'facecolor', colors(i,:));
            hold on;
        end
        hold off;
        
        title([optionText 'AUC 1-precision recall: trained with ' combinations{idxComb,1} ' and tested on ' combinations{idxComb,2} ])
        set(gca, 'XTick', 1:size(AUC,2))
        legend(legend0{:,1}, 'Location','SouthEast');
        text(1:size(AUC,2),AUC(1,:)',num2str(AUC(1,:)','%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom')
        
        drawnow;
    end
end

function [] = displayPrecisionRecall(combinations,idxComb,prec,rec, legend_c, optionText, optionalMatchingLegend)
    

    nbconf = size(prec,1);
    nbpair = size(prec,2);
    
    targetMatchingLegend = '';
    if (exist('optionalMatchingLegend','var'))
        targetMatchingLegend = optionalMatchingLegend;
    end

    if (feature('ShowFigureWindows'))
        for iconfig=1:nbconf
            
            for ipair=1:nbpair
                
                %%skip if no matching the target legend
                if (~isempty(targetMatchingLegend))
                    if (isempty(strfind(legend_c{iconfig,ipair},targetMatchingLegend)))
                        continue;
                    end
                end
                
                valX = prec{iconfig,ipair};
                valX = 1 - valX;
                valY = rec{iconfig,ipair};
                
                figure;
                plot(valX,valY);
                axis([0 1 0 1]);
                title([optionText '1-precision recall / Config: ' legend_c{iconfig,ipair} ])
            end
        end
        
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% computeDescriptors.m ends here
