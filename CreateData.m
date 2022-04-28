clc;clear all;close all;
Multy = 0;

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

%% Loop over all index and create dataset - naive version!!
%we already validate that each of this inds are sequential!
X = [];
Y = [];
for k = 1:1:length(UniqList)
    Inds = find(Device_ID_New==k); 
    NumOfWindows = floor(length(Inds)/360);
    remInds = mod(length(Inds),360);
    
    Left = reshape(RSSI_Left(Inds(1:end-remInds)),360,NumOfWindows).';
    Right = reshape(RSSI_Right(Inds(1:end-remInds)),360,NumOfWindows).';
    
    NewX = zeros(NumOfWindows,2,360);
    NewX(:,1,:) = Left;
    NewX(:,2,:) = Right;
    
    Y_image =  reshape(Num_People(Inds(1:end-remInds)),360,NumOfWindows).';

    %multi-class or binary ver:
    if Multy == 0
        Y_image = (Y_image>0);
    end
   
    Valids = all(Y_image == mean(Y_image,2),2); %Out of 85 windows, 80 valid for bin ver, and 70 for multy class ver
    NewX = NewX(Valids,:,:);
    NewY =  round(mean(Y_image(Valids,:),2));

    X = [X;NewX];
    Y = [Y;NewY];
end

X = cast(X,'single');
Y = cast(Y,"int64");

%% Split train and val
l1 = [0:1:length(Y)-1].';
l2 = [0:36:length(Y)].';
Lia = ismember(l1,l2);
l1 = l1(~Lia);
%% Save
save('Data\DataV1.mat','X','Y','l1','l2');