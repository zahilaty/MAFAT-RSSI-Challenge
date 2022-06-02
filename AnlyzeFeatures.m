clc;clear all;close all;
addpath('MatlabFunctions');
load('E:\Projects\MAFAT_RSSI\Data\DataV2_mul.mat')

%% Channels
x1 = squeeze(X(:,1,:));
x2 = squeeze(X(:,2,:));
ch1 = (x1 + x2)/2;
ch2 = abs(x1 - x2);
ch3 = [diff(ch1,1,2) zeros(length(ch1),1)];
ch4 = [diff(ch2,1,2) zeros(length(ch1),1)];

%% Sample inds
n = 10;
inds0 = find(Y==0);imax=length(inds0);sub_inds = randi(imax,n,1);sampled_inds0 = inds0(sub_inds);
inds1 = find(Y==1);imax=length(inds1);sub_inds = randi(imax,n,1);sampled_inds1 = inds1(sub_inds);
inds2 = find(Y==2);imax=length(inds2);sub_inds = randi(imax,n,1);sampled_inds2 = inds2(sub_inds);
inds3 = find(Y==3);imax=length(inds3);sub_inds = randi(imax,n,1);sampled_inds3 = inds3(sub_inds);
sampled_inds_mat = [sampled_inds0 sampled_inds1 sampled_inds2 sampled_inds3];

%% Raw Channels
Plot4ClassesSubPlots(ch1,sampled_inds_mat,'CH1',[-70 -30]);
Plot4ClassesSubPlots(ch2,sampled_inds_mat,'CH2',[0 15]);
Plot4ClassesSubPlots(ch3,sampled_inds_mat,'CH3',[-10 10]);
Plot4ClassesSubPlots(ch4,sampled_inds_mat,'CH4',[-10 10]);

%% Channels Mean
% People are blocking wifi signals (see  Mean CH1)
% People are making beamforming switching rapidly (see Mean CH3 ,4 or Var..)
x1_mean = mean(x1,2);x2_mean = mean(x2,2);
ch1_mean = mean(ch1,2);ch2_mean = mean(ch2,2);ch3_mean = mean(ch3,2);ch4_mean = mean(ch4,2);
Plot4CDF(ch1_mean,inds0,inds1,inds2,inds3,'CH1 Mean') % Mean ch1 - Great feature!
Plot4CDF(ch2_mean,inds0,inds1,inds2,inds3,'CH2 Mean') % Var ch2
Plot4CDF(ch3_mean,inds0,inds1,inds2,inds3,'CH3 Mean') % Var ch3  
Plot4CDF(ch4_mean,inds0,inds1,inds2,inds3,'CH4 Mean') % Mean ch4 - maybe with abs

%% Channels Var
% People are making more multy-path i.e. var on intensity see CH1)
% People are making beamforming switching rapidly (see CH2  Var..)
ch1_var = var(ch1,0,2);ch2_var = var(ch2,0,2);ch3_var = var(ch3,0,2);ch4_var = var(ch4,0,2);
Plot4CDF(ch1_var,inds0,inds1,inds2,inds3,'CH1 Var',[0 15]) % Var ch1 - Good feature!
Plot4CDF(ch2_var,inds0,inds1,inds2,inds3,'CH2 Var',[0 15]) % Var ch2 - BOOM!
Plot4CDF(ch3_var,inds0,inds1,inds2,inds3,'CH3 Var',[0 15]) % Var ch3 - maybe
Plot4CDF(ch4_var,inds0,inds1,inds2,inds3,'CH4 Var',[0 15]) % Var ch4 - maybe

%% Channels covariance and Correlation
% Important: covariance without deviding the variance is meaningless . see ch12 cov vs normlized cov for example(e.g.
% we need correlation)
[r, c] = size(ch2);
ch1_ZeroMean = ch1 - ch1_mean;
ch2_ZeroMean = ch2 - ch2_mean;
ch3_ZeroMean = ch3 - ch3_mean;
ch4_ZeroMean = ch4 - ch4_mean;

ch12_cov = zeros(r,1);
ch13_cov = zeros(r,1);
ch23_cov = zeros(r,1);
ch12_corr = zeros(r,2*c-1);
ch34_corr = zeros(r,2*c-1);

for i=1:r
   tmp = cov(ch1(i,:),ch2(i,:));
   ch12_cov(i,:) = tmp(2,1);
   tmp = cov(ch1(i,:),ch3(i,:));
   ch13_cov(i,:) = tmp(2,1);
   tmp = cov(ch2(i,:),ch3(i,:));
   ch23_cov(i,:) = tmp(2,1);
   ch12_corr(i,:) = xcorr(ch1_ZeroMean(i,:),ch2_ZeroMean(i,:));
   ch34_corr(i,:) = xcorr(ch3_ZeroMean(i,:),ch4_ZeroMean(i,:));
end
%Plot4CDF(ch12_cov,inds0,inds1,inds2,inds3,'CH12 Cov',[0 15]) 
Plot4CDF(ch12_cov./sqrt(ch1_var+eps)./sqrt(ch2_var+eps),inds0,inds1,inds2,inds3,'CH12 Cross Corr',[-1 1]) %nice
Plot4CDF(ch13_cov./sqrt(ch1_var+eps)./sqrt(ch3_var+eps),inds0,inds1,inds2,inds3,'CH13 Cross Corr',[-1 1]) %nice
Plot4CDF(ch23_cov./sqrt(ch2_var+eps)./sqrt(ch3_var+eps),inds0,inds1,inds2,inds3,'CH23 Cross Corr',[-1 1])  %nice
Plot4ClassesSubPlots(ch12_corr./sqrt(ch1_var+eps)./sqrt(ch2_var+eps),sampled_inds_mat,'CH12 + mean removal Corr');
Plot4ClassesSubPlots(ch34_corr./sqrt(ch3_var+eps)./sqrt(ch4_var+eps),sampled_inds_mat,'CH34 + mean removal Corr');

%% Var over correlation between channels
ch34_corr_var = var(ch34_corr,0,2);
Plot4CDF(ch34_corr_var,inds0,inds1,inds2,inds3,'CH34 Corr Var',[0 120]) % nice
Plot4PDF(ch34_corr_var,inds0,inds1,inds2,inds3,'CH34 Corr Var',[0:5:1e3],[0 120]) % nice

%% Cumsums 
ch1_mean_cumsum = cumsum((ch1-mean(ch1,2)),2); %Bad feature
ch1_var_cumsum = cumsum((ch1-mean(ch1,2)).^2,2); %Bad feature
ch2_var_cumsum = cumsum((ch2-mean(ch2,2)).^2,2);
Plot4ClassesCombined(ch1_mean_cumsum,sampled_inds_mat,'CH1 Var Cumsum')
%Plot4ClassesCombined(ch1_var_cumsum,sampled_inds_mat,'CH1 Var Cumsum')
Plot4ClassesCombined(ch2_var_cumsum,sampled_inds_mat,'CH2 Var Cumsum')

%% TBD - Bispectrum?

%% Correlation between raw channels
x1_var = var(x1,0,2);
x2_var = var(x2,0,2);
x1_ZeroMean = x1 - x1_mean;
x2_ZeroMean = x2 - x2_mean;
[r, c] = size(ch2);
x12_cov = zeros(r,1);
x12_corr = zeros(r,2*c-1);
for i=1:r
   tmp = cov(x1(i,:),x2(i,:));
   x12_cov(i,:) = tmp(2,1);
   x12_corr(i,:) = xcorr(x1_ZeroMean(i,:),x2_ZeroMean(i,:));
end
Plot4CDF(x12_cov./sqrt(x1_var+eps)./sqrt(x2_var+eps),inds0,inds1,inds2,inds3,'x12 Cov devided by vars (cross corr)',[-1 1]); %nice
Plot4ClassesSubPlots(x12_corr./sqrt(x1_var+eps)./sqrt(x2_var+eps),sampled_inds_mat,'X12 + mean removal Corr',[-1 1]); %The best so far feature!

%% Scatters - TBD - the goal is to combine several features that were good
% figure
% scatter(ch2_var(inds0),ch12_cov(inds0),2,'b');hold on
% scatter(ch2_var(inds1),ch12_cov(inds1),2,'r');hold on
% scatter(ch2_var(inds2),ch12_cov(inds2),2,'g');hold on
% scatter(ch2_var(inds3),ch12_cov(inds3),2,'k');hold on
% Mat = [ch1_var ch12_cov ch2_var].';
% figure
% plot(Mat(:,sampled_inds0),'b');hold on
% plot(Mat(:,sampled_inds1),'r');hold on
% plot(Mat(:,sampled_inds2),'g');hold on
% plot(Mat(:,sampled_inds3),'k');hold on

%% Channels PSD and FFT - bad features
ch1_pxx = periodogram(ch1.').';
ch2_pxx = periodogram(ch2.').';
ch3_pxx = periodogram(ch3.').';
ch4_pxx = periodogram(ch4.').';
% ch1_pxx = db(fftshift(fft(ch1.')).');
% ch2_pxx = db(fftshift(fft(ch2.')).');
% ch3_pxx = db(fftshift(fft(ch3.')).');
% ch4_pxx = db(fftshift(fft(ch4.')).');
x12_psd = db(fftshift(fft(x12_corr./sqrt(x1_var+eps)./sqrt(x2_var+eps))));
ch12_psd = db(fftshift(fft(ch12_corr./sqrt(ch1_var+eps)./sqrt(ch2_var+eps))));
ch34_psd = db(fftshift(fft(ch34_corr./sqrt(ch3_var+eps)./sqrt(ch4_var+eps))));
Plot4ClassesSubPlots(ch1_pxx,sampled_inds_mat,'CH1 PSD',[0 60]); %PSD ch1 
Plot4ClassesSubPlots(ch2_pxx,sampled_inds_mat,'CH2 PSD',[0 60]); %PSD ch2
Plot4ClassesSubPlots(ch3_pxx,sampled_inds_mat,'CH3 PSD',[0 60]);
Plot4ClassesSubPlots(ch4_pxx,sampled_inds_mat,'CH4 PSD',[0 60]);
Plot4ClassesSubPlots(x12_psd,sampled_inds_mat,'X12 PSD');
Plot4ClassesSubPlots(ch12_psd,sampled_inds_mat,'CH12 PSD');
Plot4ClassesSubPlots(ch34_psd,sampled_inds_mat,'CH34 PSD');

%% histogram (cant be understood)      
% edges = [0:0.25:10];
% figure;
% subplot(4,1,1);histogram(ch1_var(inds0,:),edges);
% subplot(4,1,2);histogram(ch1_var(inds1,:),edges);
% subplot(4,1,3);histogram(ch1_var(inds2,:),edges);
% subplot(4,1,4);histogram(ch1_var(inds3,:),edges);

%% TSNE ch2 (bad feature)
% Vecs = tsne(ch2);
% figure;
% plot(Vecs(inds0,1),Vecs(inds0,2),'xb');hold on;
% plot(Vecs(inds1,1),Vecs(inds1,2),'xr');hold on;
% plot(Vecs(inds2,1),Vecs(inds2,2),'xg');hold on;
% plot(Vecs(inds3,1),Vecs(inds3,2),'xk');hold on;