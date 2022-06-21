clc;clear all;close all;
addpath('MatlabFunctions');

%% Read data:
[Time, Room_Num, Device_ID, RSSI_Left, RSSI_Right, Num_People] = importfile('Data\mafat_wifi_challenge_training_set_v1');

%% Create a "categorical" Device Id List (numbers from 1 to 84 instead some big numbers)
UniqList = sort(unique(Device_ID),'ascend'); 
%The dictionary key is 1:1:84. for example: find(UniqList==92178)
Device_ID_New = zeros(size(Device_ID));
for k = 1:1:length(Device_ID)
    Device_ID_New(k) = find(UniqList==Device_ID(k));
end
assert(Device_ID(123456) == UniqList(Device_ID_New(123456)))

%%  Loop over all index and create dataset - with overlaps of 120 samples
X = [];
Y = [];
ValidsAll = [];
SampLen = 360;
Devider = 3;
Portion = SampLen/Devider;

for k = 1:1:length(UniqList)
    disp(k)
    Inds = find(Device_ID_New==k);  %where the device id appears in the excel file
       
    % Get the device Id rows (i.e. "Inds")
    NumOfWindows = floor(length(Inds)/SampLen); %the number of complete 360 samples window we have (81 for our example)
    remInds = mod(length(Inds),SampLen);

    % Extract the RSS values and num of people in room, and arrange them in *120*
    % samples window and then *360* window
    Left_phase_0 = reshape(RSSI_Left(Inds(1:end-remInds)),Portion,Devider*NumOfWindows); %120X243
    Right_phase_0 = reshape(RSSI_Right(Inds(1:end-remInds)),Portion,Devider*NumOfWindows);%120X243
    Y_image_phase_0 =  reshape(Num_People(Inds(1:end-remInds)),Portion,Devider*NumOfWindows);%120X243

    % we can go over our matrix column, and create a 360 window, until we
    % get to (243-3), and then, the 241-243 columns create the last 360
    % window
    NewNumberOfWindows = Devider*NumOfWindows-(Devider-1); %241
    Left = zeros(NewNumberOfWindows ,SampLen); %241X360 (without transpose as in the last script)
    Right = zeros(NewNumberOfWindows ,SampLen); %241X360 (without transpose as in the last script)
    Y_image = zeros(NewNumberOfWindows ,SampLen); %241X360 (without transpose as in the last script)
    
    for ind = 1:1:NewNumberOfWindows
        %first slice goes 1:3, last slice goes {Devider*NumOfWindows-(Devider-1)}:{Devider*NumOfWindows-(Devider-1) + (Devider-1)} = {241:243}
        Tmp = Left_phase_0(:,ind:(ind+(Devider-1))); 
        Left(ind,:) = Tmp(:);
        Tmp = Right_phase_0(:,ind:(ind+(Devider-1)));
        Right(ind,:) = Tmp(:);
        Tmp = Y_image_phase_0(:,ind:(ind+(Devider-1)));
        Y_image(ind,:) = Tmp(:);
    end
       
    % Allocate memory for the data
    NewX = zeros(NewNumberOfWindows ,2,SampLen); %241,2,360
    NewX(:,1,:) = Left; %241X360 -> %241,2,360
    NewX(:,2,:) = Right;
    
% Unlike before, all windows are valid, but we need to think if MAFAT
% condition (retzef of 20 samples) if true (starting from the max)
% lets do it quick and dirty
    NewY =  zeros(NewNumberOfWindows ,1); %241,1
    for ind = 1:1:(Devider*NumOfWindows-(Devider-1))
        sig = Y_image(ind,:);
        if FindIfWeHaveRetzefOf20Samles(sig,3)
            NewY(ind) = 3;
            continue
        end
        if FindIfWeHaveRetzefOf20Samles(sig,2)
            NewY(ind) = 2;
            continue
        end
        if FindIfWeHaveRetzefOf20Samles(sig,1)
            NewY(ind) = 1;
            continue
        end
        NewY(ind) = 0; %if not other condition is true   
    end
    
    Valids = all(Y_image == mean(Y_image,2),2);
    ValidsAll = [ValidsAll;Valids];
        % Extract the valid time windows, with no human movment out\in the room
%         Valids = all(Y_image == mean(Y_image,2),2); %Out of 85 windows, 80 valid for bin ver, and 70 for multy class ver
%         NewX = NewX(Valids,:,:);
%         NewY =  round(mean(Y_image(Valids,:),2));
        
    % Append to the dataset
    X = [X;NewX];
    Y = [Y;NewY];     
end

% Weights
lenY = length(Y);
ClassNumY = length(unique(Y));
Scale = ClassNumY/lenY;
w0 = sum(Y==0)*Scale;w1 = sum(Y==1)*Scale;w2 = sum(Y==2)*Scale;w3 = sum(Y==3)*Scale;
W = zeros(size(Y));
for k = 1:1:length(Y)
    if (Y(k)==0) W(k) = 1/w0; end
    if (Y(k)==1) W(k) = 1/w1; end
    if (Y(k)==2) W(k) = 1/w2; end
    if (Y(k)==3) W(k) = 1/w3; end
end

% Casting to reduce space
BigX = cast(X,'single');
BigY = cast(Y,"int64");
BigW = cast(W,"single");
clear X Y W

% Split train and val - we need to throw (Devider-1) inds from each side to avoid overlaps
l1 = [0:1:length(BigY)-1].';
l2 = [0:36*(Devider):length(BigY)].'; %every Devider*36th window will be used as val set
ToBeRemove = [];
tmp_vec = -(Devider-1):1:(Devider-1);
for k =1:1:length(l2)
    curr_ind = l2(k);
    ToBeRemove= [ToBeRemove;(tmp_vec+curr_ind)];
end
Lia = ismember(l1,ToBeRemove);
l1 = l1(~Lia);

% Save train
X = BigX(l1+1,:,:);
Y = BigY(l1+1,:,:);
W = BigW(l1+1,:,:);
V = ValidsAll(l1+1,:);
save('Data\DataV3_mul_train.mat','X','Y','W','V');

% Save val
X = BigX(l2+1,:,:);
Y = BigY(l2+1,:,:);
W = BigW(l2+1,:,:);
V = ValidsAll(l2+1,:);
save('Data\DataV3_mul_val.mat','X','Y','W','V');


%% Creating a balanced set with augmentations
% I prefer to add samples with cyclic-shift \ horizontal translation,
% becuse flip will happen any way (p=0.5..), and I want it to happend any way, 
% but what are the chances we will had a translation that would be exactly 42samples??
% it is 1/360.. so will not happen in 250 epochs..

if 1

%making more data just from the train set
X = BigX(l1+1,:,:);
Y = BigY(l1+1,:,:);
W = BigW(l1+1,:,:);

inds0 = find(Y==0);
inds1 = find(Y==1); %we need to find now, because Y is changing
inds2 = find(Y==2);
inds3 = find(Y==3); %we need to find now, because Y is changing

% For class balancing:
NumToAdd_0 = length(inds2) - length(inds0);
NumToAdd_1 = length(inds2) - length(inds1);
NumToAdd_3 = length(inds2) - length(inds3);

% Create several cyclic shift (arbitrary times interval), along the 3 dim
% (the time dim). we now we wont need more than X5 factor, so 5 shifts are  enough
X0 = X(inds0,:,:);
X1 = X(inds1,:,:);
X3 = X(inds3,:,:);

NewX0 = [circshift(X0,-42,3); circshift(X0,42,3); circshift(X0,-134,3); circshift(X0,134,3);circshift(X0,69,3)];
NewX1 = [circshift(X1,-42,3); circshift(X1,42,3); circshift(X1,-134,3); circshift(X1,134,3);circshift(X1,69,3)];
NewX3 = [circshift(X3,-42,3); circshift(X3,42,3); circshift(X3,-134,3); circshift(X3,134,3);circshift(X3,69,3)];

tmp0 = datasample(NewX0,NumToAdd_0,1);
tmp1 = datasample(NewX1,NumToAdd_1,1);
tmp3 = datasample(NewX3,NumToAdd_3,1);

% now adding them to the dataset - be carefull with the Y values!!
X = [X;tmp0;tmp1;tmp3];
tmp0 = datasample(Y(inds0),NumToAdd_0,1);
tmp1 = datasample(Y(inds1),NumToAdd_1,1);
tmp3 = datasample(Y(inds3),NumToAdd_3,1);
Y = [Y;tmp0;tmp1;tmp3];     
 
% Weights - calc them again... (should be all equal to 0.25)
lenY = length(Y);
ClassNumY = length(unique(Y));
Scale = ClassNumY/lenY;
w0 = sum(Y==0)*Scale;w1 = sum(Y==1)*Scale;w2 = sum(Y==2)*Scale;w3 = sum(Y==3)*Scale;
W = zeros(size(Y));
for k = 1:1:length(Y)
    if (Y(k)==0) W(k) = 1/w0; end
    if (Y(k)==1) W(k) = 1/w1; end
    if (Y(k)==2) W(k) = 1/w2; end
    if (Y(k)==3) W(k) = 1/w3; end
end
W = cast(W,"single");

% Split train and val - we need to throw (Devider-1) inds from each side to avoid overlaps
% l1 = [0:1:length(Y)-1].';
% l2 = [0:36*(Devider):length(Y)].'; %every Devider*36th window will be used as val set
% ToBeRemove = [];
% tmp_vec = -(Devider-1):1:(Devider-1);
% for k =1:1:length(l2)
%     curr_ind = l2(k);
%     ToBeRemove= [ToBeRemove;(tmp_vec+curr_ind)];
% end
% Lia = ismember(l1,ToBeRemove);
% l1 = l1(~Lia);

% Save
save('Data\DataV4_mul_train.mat','X','Y','W');
end


%% Creating a balanced set from 0 \ 1 perspective
if 1

%making more data just from the train set
X = BigX(l1+1,:,:);
Y = BigY(l1+1,:,:);
W = BigW(l1+1,:,:);

inds0 = find(Y==0);
inds1 = find(Y==1); %we need to find now, because Y is changing
inds2 = find(Y==2);
inds3 = find(Y==3); %we need to find now, because Y is changing

% For class balancing:
NumToAdd_0 = (length(inds2) - length(inds0)) + 2*length(inds2); %same as before, but now we need weights of [0.5,0.5/3,0.5/3,0.5/3];
NumToAdd_1 = length(inds2) - length(inds1);
NumToAdd_3 = length(inds2) - length(inds3);

% Create several cyclic shift (arbitrary times interval), along the 3 dim
% (the time dim). we now we wont need more than X5 factor, so 5 shifts are  enough
X0 = X(inds0,:,:);
X1 = X(inds1,:,:);
X3 = X(inds3,:,:);

NewX0 = [circshift(X0,-42,3); circshift(X0,42,3); circshift(X0,-134,3); circshift(X0,134,3);circshift(X0,69,3)];
NewX1 = [circshift(X1,-42,3); circshift(X1,42,3); circshift(X1,-134,3); circshift(X1,134,3);circshift(X1,69,3)];
NewX3 = [circshift(X3,-42,3); circshift(X3,42,3); circshift(X3,-134,3); circshift(X3,134,3);circshift(X3,69,3)];

tmp0 = datasample(NewX0,NumToAdd_0,1);
tmp1 = datasample(NewX1,NumToAdd_1,1);
tmp3 = datasample(NewX3,NumToAdd_3,1);

% now adding them to the dataset - be carefull with the Y values!!
X = [X;tmp0;tmp1;tmp3];
tmp0 = datasample(Y(inds0),NumToAdd_0,1);
tmp1 = datasample(Y(inds1),NumToAdd_1,1);
tmp3 = datasample(Y(inds3),NumToAdd_3,1);
Y = [Y;tmp0;tmp1;tmp3];     
 
% Weights - calc them again... (should be all equal to 0.25)
lenY = length(Y);
ClassNumY = length(unique(Y));
Scale = ClassNumY/lenY;
w0 = sum(Y==0)*Scale;w1 = sum(Y==1)*Scale;w2 = sum(Y==2)*Scale;w3 = sum(Y==3)*Scale;
W = zeros(size(Y));
for k = 1:1:length(Y)
    if (Y(k)==0) W(k) = 1/w0; end
    if (Y(k)==1) W(k) = 1/w1; end
    if (Y(k)==2) W(k) = 1/w2; end
    if (Y(k)==3) W(k) = 1/w3; end
end
W = cast(W,"single");

% Split train and val - we need to throw (Devider-1) inds from each side to avoid overlaps
% l1 = [0:1:length(Y)-1].';
% l2 = [0:36*(Devider):length(Y)].'; %every Devider*36th window will be used as val set
% ToBeRemove = [];
% tmp_vec = -(Devider-1):1:(Devider-1);
% for k =1:1:length(l2)
%     curr_ind = l2(k);
%     ToBeRemove= [ToBeRemove;(tmp_vec+curr_ind)];
% end
% Lia = ismember(l1,ToBeRemove);
% l1 = l1(~Lia);

% Save
save('Data\DataV5_mul_train.mat','X','Y','W');

end

