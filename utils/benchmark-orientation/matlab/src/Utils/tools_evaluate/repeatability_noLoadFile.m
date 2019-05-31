function [erro,repeat,corresp, match_score,matches, twi0, twi1, dout, tdout, wout, twout, sdout]= ...
        repeatability_noLoadFile(f1,f2,H, im1,im2,metric)


    %Modified from the original version to not load files
    % remember that f1 should be (x,y,a,b,c)' for each col.... image at x,y is image(y,x)
    %
    %Computes repeatability and overlap score between two lists of features
    %detected in two images.
    %   [erro,repeat,corresp,matched,matchedp]=repeatability('file1','file2','H4to1','imf1','imf2','-ko');
    %
    %IN: 
    %    file1 - file 1 with detected features         
    %    file2 - file 2 with detected features 
    %    H - file with 3x3 Homography matrix from image 1 to 2, x2=H*x1
    %        Assumes that image coordiantes are 0..width.
    %    imf1 - image file  1
    %    imf2 - image file  2
    %    common_part - flag should be set to 1 for repeatability and 0 for descriptor performance
    %
    %OUT :    erro - overlap %
    %         repeat - repeatability %
    %         corresp - nb of correspondences
    %         match_score  - matching score
    %         matches - number of correct nearest neighbour matches
    %         twi - matrix with overlap errors<50\%
    %
    %  region file format :
    %--------------------
    %descriptor_size  
    %nbr_of_regions
    %x1 y1 a1 b1 c1 d1 d2 d3 ...
    %x2 y2 a2 b2 c2 d1 d2 d3 ...
    %....
    %....
    %---------------------
    %x, y - center coordinates
    %a, b, c - ellipse parameters ax^2+2bxy+cy^2=1
    %d1 d2 d3 ... - descriptor invariants
    %if descriptor_size<=1 the descriptor is ignored


    % fprintf(1,'Reading and sorting the regions...\n');

    % [f1 s1 dimdesc1]=loadFeatures(file1);
    % [f2 s2 dimdesc2]=loadFeatures(file2);

    if ~exist('metric','var')
        metric = 'L2';
    end

    % if ~exist('common_part','var')
    % common_part = 0;
    % end

    s1 = size(f1,2);
    s2 = size(f2,2);

    if (s1*s2 > 5000*5000)
        warning(['We have ' num2str(s1) ' and ' num2str(s2) ' kp, it is a lot, so I restrain them to 5000/5000 ']);
        f1 = f1(:,1:min(5000,s1));
        f2 = f2(:,1:min(5000,s2));
        
        s1 = size(f1,2);
        s2 = size(f2,2);

    end

    dimdesc1 = size(f1,1)-5;
    dimdesc=dimdesc1;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % allocate vectors for the features
    feat1=zeros(9+dimdesc,s1);
    feat2=zeros(9+dimdesc,s2);
    feat1(1:5,1:s1)=f1(1:5,1:s1);
    feat2(1:5,1:s2)=f2(1:5,1:s2);
    if size(f1,1)>1
        feat1(10:9+dimdesc,1:s1)=f1(6:5+dimdesc,1:s1);
        feat2(10:9+dimdesc,1:s2)=f2(6:5+dimdesc,1:s2);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %project regions from one image to the other
    %%test if it is an homography or a map (i->j)
    if (all(size(H) ~= [3,3]))
        
        %separate map H and distance ratio 
        tmp = H;
        H = tmp(1:end/2,1:end);
        ratioM = tmp(end/2+1:end,1:end);
        
        
        %use fix scale for all
        %a = c = 1./(s*s);
        %     fixscale = 10;
        %     a = 1/(fixscale*fixscale);
        %     feat1(3,:) = a;feat1(5,:) = a;feat1(4,:) = 0;
        %     feat2(3,:) = a;feat2(5,:) = a;feat2(4,:) = 0;
        




        fprintf(1,'Projecting 1 to 2...');
        [feat1 ,feat1t, ~]=project_regions(feat1',eye(3));
        fprintf(1,'and 2 to 1...\n');
        [feat2,feat2t ,~]=project_regions(feat2',eye(3));
        
        %%do the mapping now from 2 to 1
        idxp1 = sub2ind(size(im1),round(feat2(:,2)),round(feat2(:,1)));

        idxp2 = H(idxp1); ratios = ratioM(idxp1);
        feat2NoProjection = idxp2==-1;
        feat2YesProjection = idxp2~=-1;
        
        % IMPORTANT! below removing is very harmful! will cause twi to
        % be in different dimensions and all sorts of wrong things will
        % happen! Instead, remember the indices where we SHOULD COMPUTE
        
        % %remove the points which do not project on im1
        % feat2(feat2NoProjection,:) = [];
        % feat2t(feat2NoProjection,:) = [];
        % ratios(feat2NoProjection,:) = [];
        % idxp2(feat2NoProjection) = [];
        
        [p2y,p2x] = ind2sub(size(im1),idxp2);
        
        % display(size(feat2t(feat2YesProjection,3)));
        % display(size(ratios(feat2YesProjection)));
        
        if sum(feat2YesProjection) > 0
            feat2t(feat2YesProjection,1) = p2x(feat2YesProjection);
            feat2t(feat2YesProjection,2) = p2y(feat2YesProjection);
            feat2t(feat2YesProjection,3) = feat2t(feat2YesProjection,3)./(ratios(feat2YesProjection).^2);
            feat2t(feat2YesProjection,5) = feat2t(feat2YesProjection,5)./(ratios(feat2YesProjection).^2);
            feat2t(feat2YesProjection,4) = feat2t(feat2YesProjection,4)./(ratios(feat2YesProjection).^2);
        end

        % Instead of deleting, assign non-sense values
        feat2(feat2NoProjection,:) = -1000;
        feat2t(feat2NoProjection,:) = -1000;
        ratios(feat2NoProjection,:) = -1000;
        idxp2(feat2NoProjection) = -1000;
        
        % diplay('okay until here...');
        
        % if false
        %       idxpt = 55;
        %       imshow(im1); drawellipse([feat2t(idxpt,3) feat2t(idxpt,4); feat2t(idxpt,4) feat2t(idxpt,5) ], feat2t(idxpt,1),feat2t(idxpt,2),'y');
        %       figure;imshow(im2); drawellipse([feat2(idxpt,3) feat2(idxpt,4); feat2(idxpt,4) feat2(idxpt,5) ], feat2(idxpt,1),feat2(idxpt,2),'y');
        
        %       imshow(im2);
        %     hold on;plot(feat2(1,1),feat2(1,2),'+')
        %      figure;
        %      imshow(im1);hold on;plot(feat2t(1,2),feat2t(1,1),'+')
        
        %     idxp1 = find(H~=-1);
        %     selection = randperm(size(idxp1,1),50);
        %     idxp1 = idxp1(selection);
        %     idxp2 = H(idxp1);
        %     imgi = (im2);
        %     imgj = (im1);
        %     [p1y,p1x] = ind2sub(size(imgi),idxp1);
        %     [p2y,p2x] = ind2sub(size(imgi),idxp2);

        %     figure; ax = axes;
        %     showMatchedFeatures(imgi,imgj,[p1x,p1y],[p2x,p2y],'montage','Parent',ax);
        %     title(ax, 'Candidate point matches');
        %     legend(ax, 'Matched points 1','Matched points 2');
        % end
    else
        HI=H(:, 1:3);
        H=inv(HI);
        fprintf(1,'Projecting 1 to 2...');
        [feat1 ,feat1t, ~]=project_regions(feat1',HI);
        fprintf(1,'and 2 to 1...\n');
        [feat2,feat2t ,~]=project_regions(feat2',H);
        
        % For compatibility
        feat2YesProjection = true(size(feat2,1),1);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %if false
    % if common_part==1 
    %     fprintf(1,'Removing features from outside of the common image part...\n');
    %     im1x=size(im1);
    %     im1y=im1x(1);
    %     im1x=im1x(2);
    %     im2x=size(im2);
    %     im2y=im2x(1);
    %     im2x=im2x(2);
    %     ind=find((feat1(:,1)+feat1(:,8))<im1x & (feat1(:,1)-feat1(:,8))>0 & (feat1(:,2)+feat1(:,9))<im1y & (feat1(:,2)-feat1(:,9))>0);
    %     feat1=feat1(ind,:);
    %     %feat1t=feat1t(ind,:);
    %     ind=find((feat1t(:,1)+feat1t(:,8))<im2x & (feat1t(:,1)-feat1t(:,8))>0 & (feat1t(:,2)+feat1t(:,9))<im2y & (feat1t(:,2)-feat1t(:,9))>0);
    %     feat1=feat1(ind,:);
    %     %feat1t=feat1t(ind,:);
    %     %scales1=scales1(ind);
    % 
    %     ind=find((feat2(:,1)+feat2(:,8))<im2x & (feat2(:,1)-feat2(:,8))>0 & (feat2(:,2)+feat2(:,9))<im2y & (feat2(:,2)-feat2(:,9))>0);
    %     feat2t=feat2t(ind,:);
    %     %feat2=feat2(ind,:);
    %     ind=find((feat2t(:,1)+feat2t(:,8))<im1x & (feat2t(:,1)-feat2t(:,8))>0 & (feat2t(:,2)+feat2t(:,9))<im1y & (feat2t(:,2)-feat2t(:,9))>0);
    %     feat2t=feat2t(ind,:);
    %     %feat2=feat2(ind,:);
    %     %scales2=scales2(ind);
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 
    %     fprintf(1,'nb of regions in common part in image1 %d.\n',size(feat1,1));
    %     fprintf(1,'nb of regions in common part in image2 %d.\n',size(feat2t,1));
    % end 
    %end

    sf=min([size(feat1,1) size(feat2t,1)]);

    feat1=feat1';
    %feat1t=feat1t';
    feat2t=feat2t';
    %feat2=feat2';

    fprintf(1,'Computing overlap error & selecting one-to-one correspondences: ');
    tic;
    %c_eoverlap is a C implementation to compute the overlap error.


    % Initialize with basically saying we have no match (both GT and desc wise)
    wout = 100 * ones(size(feat2t,2),size(feat1,2));
    twout = 100 * ones(size(feat2t,2),size(feat1,2));
    dout = 1000000 * ones(size(feat2t,2),size(feat1,2));
    tdout = 1000000 * ones(size(feat2t,2),size(feat1,2));
    sdout = zeros(size(feat2t,2),size(feat1,2));
    
    % Compute if we have at least one feat matched
    if sum(feat2YesProjection) > 0
        common_part = 0;
        tmp_wout = [];
    
        if strcmp(metric,'Hamming') == 1
            [tmp_wout, tmp_twout, tmp_dout, tmp_tdout, tmp_sdout]=c_eoverlap_hamming(feat1,feat2t(:,feat2YesProjection),common_part);
        end
        if strcmp(metric,'L2') == 1
            [tmp_wout, tmp_twout, tmp_dout, tmp_tdout, tmp_sdout]=c_eoverlap_l2(feat1,feat2t(:,feat2YesProjection),common_part);
        end
        if strcmp(metric,'EMD') == 1
            [tmp_wout, tmp_twout, tmp_dout, tmp_tdout, tmp_sdout]=c_eoverlap_emd(feat1,feat2t(:,feat2YesProjection),common_part);
        end
        if strcmp(metric,'L1') == 1
            [tmp_wout, tmp_twout, tmp_dout, tmp_tdout, tmp_sdout]=c_eoverlap_l1(feat1,feat2t(:,feat2YesProjection),common_part);
        end
        if strcmp(metric,'sGLOH') == 1
            [tmp_wout, tmp_twout, tmp_dout, tmp_tdout, tmp_sdout]=c_eoverlap_sGLOH(feat1,feat2t(:,feat2YesProjection),common_part);
        end
        
        if (sum(feat2YesProjection)>0 && isempty(tmp_wout))
            error(['Metric is wrong or empty: Metric = ', metric]);
        end
        
        % Re-scale wout to proper size
        wout(feat2YesProjection,:) = tmp_wout;
        twout(feat2YesProjection,:) = tmp_twout;
        dout(feat2YesProjection,:) = tmp_dout;
        tdout(feat2YesProjection,:) = tmp_tdout;
        sdout(feat2YesProjection,:) = tmp_sdout;
        
    end
    
    % display(sprintf('feat2t = %d, %d   feat1 = %d, %d    tmp_wout = %d,%d\n',size(feat2t,1),size(feat2t,2),size(feat1,1),size(feat1,2),size(tmp_wout,1),size(tmp_wout,2)));
    

    t=toc;
    fprintf(1,' %.1f sec.\n',t);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    erro=[10:10:60];
    corresp=zeros(1,6);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %compute the number of correspondences
    for i=1:6,
        wi=(wout<erro(i));
        corresp(i)=sum(sum(wi));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    repeat=100*corresp/sf;


    fprintf(1,'\noverlap error: ');
    fprintf(1,'%.1f ',erro);
    fprintf(1,'\nrepeatability: ');
    fprintf(1,'%.1f ',repeat);
    fprintf(1,'\nnb of correspondences: ');
    fprintf(1,'%.0f ',corresp);
    fprintf(1,'\n');

    % match_overlap=40;
    % if common_part==0
    match_overlap=50;
    % end

    fprintf(1,'Matching with the descriptor for the overlap error < %d%%\n',match_overlap);
    twi0=(twout<match_overlap);
    twi1=(wout<match_overlap);


    dx=(dout<1000000).*(twi1);
    matches=sum(sum(dx));
    match_score=100*matches/sf;

    fprintf(1,'\nMatching score  %0.1f, nb of correct matches %.1f.\n',match_score,matches);



end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [feat,featp,scales]=project_regions(feat,H)

    s=size(feat);
    s1=s(1);

    featp=feat;
    scales=zeros(1,s1);

    for c1=1:s1,%%%%%%%%%%%%%%%%%
                %feat(c1,3:5)=(1/25)*feat(c1,3:5);
        Mi1=[feat(c1,3) feat(c1,4);feat(c1,4) feat(c1,5)];

        %compute affine transformation
        [v1 e1]=eig(Mi1);
        d1=(1/sqrt(e1(1))); 
        d2=(1/sqrt(e1(4))); 
        sc1=sqrt(d1*d2);
        feat(c1,6)=d1;
        feat(c1,7)=d2; 
        scales(c1)=sqrt(feat(c1,6)*feat(c1,7));

        %bounding box
        feat(c1,8) = sqrt(feat(c1,5)/(feat(c1,3)*feat(c1,5) - feat(c1,4)^2));
        feat(c1,9) = sqrt(feat(c1,3)/(feat(c1,3)*feat(c1,5) - feat(c1,4)^2));


        Aff=getAff(feat(c1,1),feat(c1,2),sc1, H);

        %project to image 2
        l1=[feat(c1,1),feat(c1,2),1];
        l1_2=H*l1';
        l1_2=l1_2/l1_2(3);
        featp(c1,1)=l1_2(1);
        featp(c1,2)=l1_2(2);
        BMB=inv(Aff*inv(Mi1)*Aff');
        [v1 e1]=eig(BMB);
        featp(c1,6)=(1/sqrt(e1(1)));
        featp(c1,7)=(1/sqrt(e1(4))); 
        featp(c1,3:5)=[BMB(1) BMB(2) BMB(4)];
        %bounding box in image 2
        featp(c1,8) = sqrt(featp(c1,5)/(featp(c1,3)*featp(c1,5) - featp(c1,4)^2));
        featp(c1,9) = sqrt(featp(c1,3)/(featp(c1,3)*featp(c1,5) - featp(c1,4)^2));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Aff=getAff(x,y,sc,H)
    h11=H(1);
    h12=H(4);
    h13=H(7);
    h21=H(2);
    h22=H(5);
    h23=H(8);
    h31=H(3);
    h32=H(6);
    h33=H(9);
    fxdx=h11/(h31*x + h32*y +h33) - (h11*x + h12*y +h13)*h31/(h31*x + h32*y +h33)^2;
    fxdy=h12/(h31*x + h32*y +h33) - (h11*x + h12*y +h13)*h32/(h31*x + h32*y +h33)^2;

    fydx=h21/(h31*x + h32*y +h33) - (h21*x + h22*y +h23)*h31/(h31*x + h32*y +h33)^2;
    fydy=h22/(h31*x + h32*y +h33) - (h21*x + h22*y +h23)*h32/(h31*x + h32*y +h33)^2;

    Aff=[fxdx fxdy;fydx fydy];
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

