clc;clear all;close all;
load('E:\Projects\MAFAT_RSSI\Data\DataV2_mul.mat')
%%
ch1 = squeeze(X(:,1,:) + X(:,2,:))/2;
ch2 = abs(squeeze(X(:,1,:) - X(:,2,:)));
ch3 = diff(ch1,1,2);
ch4 = diff(ch2,1,2);

%% Sample inds
n = 10;
inds0 = find(Y==0);imax=length(inds0);sub_inds = randi(imax,n,1);sampled_inds0 = inds0(sub_inds);
inds1 = find(Y==1);imax=length(inds1);sub_inds = randi(imax,n,1);sampled_inds1 = inds1(sub_inds);
inds2 = find(Y==2);imax=length(inds2);sub_inds = randi(imax,n,1);sampled_inds2 = inds2(sub_inds);
inds3 = find(Y==3);imax=length(inds3);sub_inds = randi(imax,n,1);sampled_inds3 = inds3(sub_inds);
sampled_inds_mat = [sampled_inds0 sampled_inds1 sampled_inds2 sampled_inds3];

%% Raw Channels
Plot4Classes(ch1,sampled_inds_mat,[-70 -30],'CH1');
Plot4Classes(ch2,sampled_inds_mat,[0 15],'CH2');
Plot4Classes(ch3,sampled_inds_mat,[-10 10],'CH3');
Plot4Classes(ch4,sampled_inds_mat,[-10 10],'CH4');

%% Channels Mean
% People are blocking wifi signals (see  Mean CH1)
% People are making beamforming switching rapidly (see Mean CH3 ,4 or Var..)
ch1_mean = mean(ch1,2);ch2_mean = mean(ch2,2);ch3_mean = mean(ch3,2);ch4_mean = mean(ch4,2);
Plot4CDF(ch1_mean,inds0,inds1,inds2,inds3,'CH1 Mean') % Mean ch1 - Great feature!
Plot4CDF(ch2_mean,inds0,inds1,inds2,inds3,'CH2 Mean') % Var ch2
Plot4CDF(ch3_mean,inds0,inds1,inds2,inds3,'CH3 Mean') % Var ch3  
Plot4CDF(ch4_mean,inds0,inds1,inds2,inds3,'CH4 Mean') % Mean ch4 - maybe

%% Channels Var
% People are making more multy-path i.e. var on intensity see CH1)
% People are making beamforming switching rapidly (see CH2  Var..)
ch1_var = var(ch1,0,2);ch2_var = var(ch2,0,2);ch3_var = var(ch3,0,2);ch4_var = var(ch4,0,2);
Plot4CDF(ch1_var,inds0,inds1,inds2,inds3,'CH1 Var',[0 15]) % Var ch1 - Great feature!
Plot4CDF(ch2_var,inds0,inds1,inds2,inds3,'CH2 Var',[0 15]) % Var ch2 - Great feature!
Plot4CDF(ch3_var,inds0,inds1,inds2,inds3,'CH3 Var',[0 15]) % Var ch3 - maybe
Plot4CDF(ch4_var,inds0,inds1,inds2,inds3,'CH4 Var',[0 15]) % Var ch4 - maybe

%% TBD %% Try  psd \ FFT corr? 
[r, c] = size(ch2);
New = zeros(r,2*c-1);
for i=1:r
  New(i,:) = xcorr(ch2(i,:),ch2(i,:));
end
Plot4Classes(New,sampled_inds_mat,[-70 1e6],'CH1');



function [] = Plot4Classes(Feature_2d,sampled_inds_mat,ylims,title_str)
    sampled_inds0 = sampled_inds_mat(:,1);sampled_inds1 = sampled_inds_mat(:,2);
    sampled_inds2 = sampled_inds_mat(:,3);sampled_inds3 = sampled_inds_mat(:,4);
    figure;
    subplot(4,1,1);plot(Feature_2d(sampled_inds0,:).','b');hold on;ylim(ylims)
    subplot(4,1,2);plot(Feature_2d(sampled_inds1,:).','r');hold on;ylim(ylims)
    subplot(4,1,3);plot(Feature_2d(sampled_inds2,:).','g');hold on;ylim(ylims)
    subplot(4,1,4);plot(Feature_2d(sampled_inds3,:).','k');hold on;ylim(ylims)
    sgtitle(title_str);grid on;
end

function [] = Plot4CDF(Feature_1d,inds0,inds1,inds2,inds3,title_str,xlims)
    figure
    [f,x] = ecdf(Feature_1d(inds0));plot(x,f,'b');hold on
    [f,x] = ecdf(Feature_1d(inds1));plot(x,f,'r');hold on
    [f,x] = ecdf(Feature_1d(inds2));plot(x,f,'g');hold on
    [f,x] = ecdf(Feature_1d(inds3));plot(x,f,'k');hold on
    if nargin == 7
        xlim(xlims);
    end
    title(title_str);grid on;legend('0','1','2','3');
end

%% Old
% edges = [0:0.25:10];
% figure;
% subplot(4,1,1);histogram(ch1_var(inds0,:),edges);
% subplot(4,1,2);histogram(ch1_var(inds1,:),edges);
% subplot(4,1,3);histogram(ch1_var(inds2,:),edges);
% subplot(4,1,4);histogram(ch1_var(inds3,:),edges);

%% TSNE ch2
% Vecs = tsne(ch2);
% inds0 = find(Y==0);
% inds1 = find(Y==1);
% inds2 = find(Y==2);
% inds3 = find(Y==3);
% figure;
% plot(Vecs(inds0,1),Vecs(inds0,2),'xb');hold on;
% plot(Vecs(inds1,1),Vecs(inds1,2),'xr');hold on;
% plot(Vecs(inds2,1),Vecs(inds2,2),'xg');hold on;
% plot(Vecs(inds3,1),Vecs(inds3,2),'xk');hold on;