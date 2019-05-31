function runme

% features and descriptors
% Mikolajczyk convention, except for the file header
% first image
[pts0,desc0]=get_pts('graf/graf1_sgloh.txt');
% second image
[pts1,desc1]=get_pts('graf/graf3_sgloh.txt');

% distance matrix
% the match distance between the i-th keypoint of pt0 and 
% the j-th keypoint of pts1 distance_matrix(i,j)
% the 'ms' file can be changed from the .ini file, under "ms_file"
dist_matrix=get_dist('graf/graf_ms');

% nn match
% i-th match: pts0(nn(i,1),[1 2]) pts1(nn(i,2),[1 2])
% and the match value is nn(i,3)
nn=get_match('graf/graf_nn.txt');

% nnr match
% the match value is the inverse with respect to Lowe convention 
nnr=get_match('graf/graf_nnr.txt');

disp('Done...');

function r=get_dist(descfile)

fid=fopen(descfile,'r');
d=fread(fid,2,'int');
m=fread(fid,[d(1) d(2)],'float');
fclose(fid);
r=m';

function r=get_match(descfile)

fid=fopen(descfile,'r');
l=fscanf(fid,'%d',1);
d=fscanf(fid,'%f',[3 l]);
fclose(fid);
r=d';
r(:,[1 2])=r(:,[2 1])+1;

function [pts,desc]=get_pts(descfile)

fid=fopen(descfile,'r');
l=fscanf(fid,'%d',2);
n=l(1);
m=l(2);
d=fscanf(fid,'%f',[n+5 m]);
fclose(fid);
d=d';
pts=d(:,1:5);
desc=d(:,6:end);
