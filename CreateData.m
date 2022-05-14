clc;clear all;close all;
Multy = 1;

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

%% Loop over all index and create dataset - naive version without overlaps
if 0
    %we already validate that each of this inds are sequential!
    X = [];
    Y = [];
    for k = 1:1:length(UniqList)
        % Get the device Id rows (i.e. "Inds")
        Inds = find(Device_ID_New==k);  %where the device id appears in the excel file
        NumOfWindows = floor(length(Inds)/360); %the number of complete 360 samples window we have
        remInds = mod(length(Inds),360);

        % Extract the RSS values and num of people in room, and arrange them in 360 samples windows
        Left = reshape(RSSI_Left(Inds(1:end-remInds)),360,NumOfWindows).';
        Right = reshape(RSSI_Right(Inds(1:end-remInds)),360,NumOfWindows).';
        Y_image =  reshape(Num_People(Inds(1:end-remInds)),360,NumOfWindows).';

        % Allocate memory for the data
        NewX = zeros(NumOfWindows,2,360);
        NewX(:,1,:) = Left;
        NewX(:,2,:) = Right;

        %Choose multi-class or binary ver:
        if Multy == 0
            Y_image = (Y_image>0);
        end

        % Extract the valid time windows, with no human movment out\in the room
        Valids = all(Y_image == mean(Y_image,2),2); %Out of 85 windows, 80 valid for bin ver, and 70 for multy class ver
        NewX = NewX(Valids,:,:);
        NewY =  round(mean(Y_image(Valids,:),2));

        % Append to the dataset
        X = [X;NewX];
        Y = [Y;NewY];
    end
    
    % Casting to reduce space
    X = cast(X,'single');
    Y = cast(Y,"int64");

    % Split train and val
    l1 = [0:1:length(Y)-1].';
    l2 = [0:36:length(Y)].'; %every 36th window will be used as val set
    Lia = ismember(l1,l2);
    l1 = l1(~Lia);

    % Save
    save('Data\DataV1_mul.mat','X','Y','l1','l2');
end




%%  Loop over all index and create dataset - with overlaps of 120 samples
if 1
X = [];
Y = [];
for k = 1:1:length(UniqList)
    
    Inds = find(Device_ID_New==k);  %where the device id appears in the excel file
       
    % Get the device Id rows (i.e. "Inds")
    NumOfWindows = floor(length(Inds)/360); %the number of complete 360 samples window we have
    remInds = mod(length(Inds),360);

    % Extract the RSS values and num of people in room, and arrange them in *120*
    % samples window and then *360* window
    Left_phase_0 = reshape(RSSI_Left(Inds(1:end-remInds)),120,3*NumOfWindows).';
    Right_phase_0 = reshape(RSSI_Right(Inds(1:end-remInds)),120,3*NumOfWindows).';
    Y_image_phase_0 =  reshape(Num_People(Inds(1:end-remInds)),120,3*NumOfWindows).';
    
    % now, from each 360 window, we can create 3 windows, except the last
    % window, in which we can create only 1: 3*(n-1) + 1 = 3*n-2. After
    % this block, it is the same as above, except the train\val split
    Left = zeros(3*NumOfWindows-2,360);
    Right = zeros(3*NumOfWindows-2,360);
    Y_image = zeros(3*NumOfWindows-2,360);
    for ind = 1:1:(3*NumOfWindows-2)
        Left(ind,:) = [Left_phase_0(ind,:) Left_phase_0(ind+1,:) Left_phase_0(ind+2,:)];
        Right(ind,:) = [Right_phase_0(ind,:) Right_phase_0(ind+1,:) Right_phase_0(ind+2,:)];
        Y_image(ind,:) = [Y_image_phase_0(ind,:) Y_image_phase_0(ind+1,:) Y_image_phase_0(ind+2,:)];
    end
    
    % Allocate memory for the data
    NewX = zeros(3*NumOfWindows-2,2,360);
    NewX(:,1,:) = Left;
    NewX(:,2,:) = Right;

    %Choose multi-class or binary ver:
    if Multy == 0
        Y_image = (Y_image>0);
    end
    
     % Extract the valid time windows, with no human movment out\in the room
    Valids = all(Y_image == mean(Y_image,2),2); %Out of 85 windows, 80 valid for bin ver, and 70 for multy class ver
    NewX = NewX(Valids,:,:);
    NewY =  round(mean(Y_image(Valids,:),2));
    
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
X = cast(X,'single');
Y = cast(Y,"int64");
W = cast(W,"single");

% Split train and val - we need to throw 2 inds from each side to avoid overlaps
l1 = [0:1:length(Y)-1].';
l2 = [0:36*3:length(Y)].'; %every 3*36th window will be used as val set
ToBeRemove = sort([(l2-2);(l2-1);l2;(l2+1);(l2+2)]);
Lia = ismember(l1,ToBeRemove);
l1 = l1(~Lia);

% Save
%save('Data\DataV2_mul.mat','X','Y','W','l1','l2');

end