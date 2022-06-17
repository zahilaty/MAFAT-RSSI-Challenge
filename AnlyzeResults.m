clc;clear all;close all;

%% Load or drag
load('LossVals.mat')

%% Binary accuracy
cf_matrix = double(cf_matrix_1);
Bin_classifier_cf_mat = [cf_matrix(1,1) sum(cf_matrix(1,2:end)); sum(cf_matrix(2:end,1)) sum(sum(cf_matrix(2:end,2:end)))];
Bin_acuracy = trace(Bin_classifier_cf_mat)/sum(sum(Bin_classifier_cf_mat));
disp(Bin_acuracy)

%% Plot learning curves
figure;
plot(Costs);hold on;
plot(Costs_val);hold on;
plot(Costs_val_weighted); 
ylim([0.2 1.2]);grid on

%% Plot learning curves for AUC
figure
subplot(2,1,1)
plot(Costs,'b');ylim([0 0.09]);grid on
subplot(2,1,2)
plot(Score_val,'r');ylim([0 1]);grid on

%% Plot smooth learning curves
figure;
tmp1 = reshape(Costs,100,[]);
tmp2 = reshape(Costs_val,100,[]);
tmp3 = reshape(Costs_val_weighted,100,[]);
plot(mean(tmp1,1));hold on;
plot(mean(tmp2,1));hold on;
plot(mean(tmp3,1));hold on;
ylim([0.2 1.2]);grid on

 
%% Ensambles
figure
plot(Ensemble_out);hold on
plot(targets_val,'--k')

Errors = Ensemble_out-targets_val;
Error = outputs_val-targets_val;

mean(abs(Errors(:)))
mean(abs(Error(:)))

figure
edges = [0:0.25:3.5];
subplot(2,1,1);histogram(abs(Errors(:)),edges,'Normalization','probability');grid on
subplot(2,1,2);histogram(abs(Error),edges,'Normalization','probability');grid on

figure
[f,x] = ecdf(abs(Errors(:)));plot(x,f,'b');hold on
[f,x] = ecdf(abs(Error));plot(x,f,'r');hold on

predictor_ref = round(outputs_val);
predictor_1 = round(mean(Ensemble_out,2));
predictor_2 = round(median(Ensemble_out,2));

Accuracy_ref = sum(predictor_ref==targets_val)/length(targets_val)
Accuracy_pred_1 = sum(predictor_1==targets_val)/length(targets_val)
Accuracy_pred_2 = sum(predictor_2==targets_val)/length(targets_val)

Error_ref = norm( predictor_ref-targets_val ,1)
Error_pred_1 = norm( predictor_1-targets_val , 1)
Error_pred_2 = norm( predictor_2-targets_val , 1)

%% What is the best way to create bin classfier from results?
% net.cpu()
% outputs_val = net(val_samples.cpu())
% sio.savemat('tmp.mat',{'outputs_val':outputs_val.detach().numpy()})
%load('tmp.mat')
Bin_targets_val = (targets_val>0.99);
outputs_val = round(outputs_val*1000)/1000;

%opt 1: 
prob_preds = outputs_val./sum(outputs_val,2);
prob_preds = sum(prob_preds(:,2:end),2); %probabilty of 1 or 2 or 3
[X,Y,T,AUC] = perfcurve(Bin_targets_val,prob_preds,1)

%opt 2:
prob_preds = outputs_val./sum(outputs_val,2);
prob_preds = prob_preds*[0;1;2;3];
prob_preds(prob_preds>1) = 1;
[X,Y,T,AUC] = perfcurve(Bin_targets_val,prob_preds,1)

%% Debugg the problem with predication of 3 people
%load('tmp.mat')
A = outputs_val./sum(outputs_val,2);
A = round(A*1000)/1000;
B = A*[0;1;2;3];

%% Feature scaling code?
%A = torch.squeeze(val_samples[17,:,:,0]).cpu().numpy()
%sio.savemat('tmp.mat',{'A':A})

%%
s1 = squeeze(X(111,1,:));
s2 = squeeze(X(111,2,:));
mean_s = mean([s1;s2]);
FlipedAroundMean1 = 2*mean_s -s1;
FlipedAroundMean2 = 2*mean_s -s2;
figure
subplot(2,1,1)
plot(s1,'b');hold on
plot(s2,'r');hold on
plot([1:360],ones(360,1)*mean_s,'--k');hold on
subplot(2,1,2)
plot(FlipedAroundMean1,'b');hold on
plot(FlipedAroundMean2,'r');hold on
plot([1:360],ones(360,1)*mean_s,'--k');hold on