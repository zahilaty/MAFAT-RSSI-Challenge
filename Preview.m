clc;clear all;close all;
addpath('MatlabFunctions');

%% Read data:
[Time, Room_Num, Device_ID, RSSI_Left, RSSI_Right, Num_People] = importfile('Data\mafat_wifi_challenge_training_set_v1');

%% Create a "categorical" Device Id List (numbers from 1 to 84 instead some big numbers)
UniqList = sort(unique(Device_ID),'ascend'); 
%The dictionary key is 1:1:84
Device_ID_New = zeros(size(Device_ID));
for k = 1:1:length(Device_ID)
    Device_ID_New(k) = find(UniqList==Device_ID(k));
end
assert(Device_ID(123456) == UniqList(Device_ID_New(123456)))

%% Figure 1 - the relations between the vecs
x1 = 1;%950000
x2 = length(Time);%1050000
figure
subplot(4,1,1)
%plot([0;diff(Time)]);
plot(Time);xlim([x1 x2]);grid on;xlabel('Ind');title('Time')
subplot(4,1,2)
plot(Device_ID_New);xlim([x1 x2]);grid on;xlabel('Ind');title('Device ID')
subplot(4,1,3)
plot(Num_People);xlim([x1 x2]);grid on;xlabel('Ind');title('Num of People')
subplot(4,1,4)
plot(Room_Num);xlim([x1 x2]);grid on;xlabel('Ind');title('Room Num')


%% Figure 2 - Basic 1D histograms
figure
subplot(4,1,1)
histogram(Room_Num);grid on;title('Room Num Histogram')
subplot(4,1,2)
histogram(Device_ID_New);grid on;title('Device ID Histogram')
subplot(4,1,3)
histogram(Num_People);grid on;title('Num of People Histogram')

%% Figure 3 - 2D histogram 
RoomEdges = [-Inf 0.5:1:6.5 Inf].';
DeviceEdges = [-Inf 0.5:1:84.5 Inf].';
figure
hist3([Room_Num Device_ID_New],'Edges',{RoomEdges DeviceEdges},'CDataMode','auto','FaceColor','interp')
%histogram2(Room_Num,Device_ID_New,RoomEdges,DeviceEdges)
xlabel('Room num');ylabel('Device ID');

%% Figure 4 - 2D histogram 
HumansEdges = [-Inf -0.5:1:3.5 Inf].';
DeviceEdges = [-Inf 0.5:1:84.5 Inf].';
figure
hist3([Num_People Device_ID_New],'Edges',{HumansEdges DeviceEdges},'CDataMode','auto','FaceColor','interp')
%histogram2(Room_Num,Device_ID_New,RoomEdges,DeviceEdges)
xlabel('Num of people num');ylabel('Device ID');

%% A list of each of the rooms each device saw
Rooms = {};
NumOfPeople = {};
HowManySamplesWeHave = zeros(size(UniqList));
for k = 1:1:length(UniqList)
    Inds = find(Device_ID_New==k);
    HowManySamplesWeHave(k) = length(Inds);
    Rooms{k} = unique(Room_Num(Inds));
    NumOfPeople{k} = unique(Num_People(Inds));
%     if length(unique(Room_Num(Inds))) > 1
%         disp(k)        %for debug
%     end
    %check that all inds of device are one after the another:
    assert(all(diff(Inds) == 1));
    
end