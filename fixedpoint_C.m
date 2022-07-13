close all;
clear all;
%%
%读取文件
name = 'BVP.csv';% 202112101110 202112081640
plot_range = 3100:3299;
%%
data = xlsread(name);
ppgstarttime = data(1);
ppg = data(9537:9537+64*3787);
new_ppg = resample(ppg, 25,64)';
figure;
plot(new_ppg(plot_range));
title('原始ppg');
%%
data_acc = xlsread('ACC.csv');
acc = data_acc(4769:4769+32*3787,:);
acc = resample(acc, 25,32);

new_acc = [];
for i =1:length(acc)
    new_acc(i) = sqrt(acc(i,1)^2+acc(i,2)^2+acc(i,3)^2);
end
figure;
plot(new_acc(plot_range));
title('原始acc');

%%
H10filename = ['HRband.txt'];

fileID = fopen(H10filename);
data2 = textscan(fileID, '%s'); 
fclose(fileID);
data2 = data2{1};
polarHR_data = [];
for i =1:length(data2)
    temp = data2{i};
    if strcmp(temp, 'Measurement:')
        polarHR_data = [polarHR_data str2num(data2{i+1})];
    end
end

%%
mean_new_ppg = [];
for i = 1:length(new_ppg)
    mean_new_ppg(i) = mean(new_ppg(max(1, i-3):min(length(new_ppg), i+3)));
end

new_ppg = new_ppg-mean_new_ppg;

figure;
plot(new_ppg(plot_range));
title('去均值后ppg');
%%  
mean_new_ppg = [];
for i = 1:length(new_ppg)
    mean_new_ppg(i) = mean(new_ppg(max(1, i-1):min(length(new_ppg), i+1)));
end

new_ppg = mean_new_ppg;

figure;
plot(new_ppg(plot_range));
title('均值ppg');

% pause()；
%%
mean_new_acc = [];
for i = 1:length(new_acc)
    mean_new_acc(i) = mean(new_acc(max(1, i-3):min(length(new_acc), i+3)));
end
new_acc = new_acc-mean_new_acc;

figure;
plot(new_acc(plot_range));
title('去均值后acc');
%%
mean_new_acc = [];
for i = 1:length(new_acc)
    mean_new_acc(i) = mean(new_acc(max(1, i-1):min(length(new_acc), i+1)));
end

new_acc = mean_new_acc;

figure;
plot(new_acc(plot_range));
title('均值acc');

%%
K = 100;
ppgstart = 1;
ppgend = K;
fS = 25;
Nfft=1500;
df=fS/Nfft;
k_L = floor(0.5/df);
k_R = ceil(3/df);
bpm = [];
i = 1;
ind_list = [];
stdacc_list = [];

% figure;

Hwin = hanning(K)';
while ppgend<length(new_ppg)
    
x = new_ppg(ppgstart:ppgend);
x = (x-mean(x))/max(abs((x-mean(x))));

% x = x.*Hwin;

acc_slice = new_acc(ppgstart:ppgend);
acc_slice = (acc_slice-mean(acc_slice))/max(abs((acc_slice-mean(acc_slice))));

% acc_slice = acc_slice.*Hwin;

N = k_R-k_L+1;
a= zeros(1, N);
b= zeros(1, N);
c= zeros(1, N);
axx= zeros(1, N);
bxx= zeros(1, N);
N = Nfft;


for k=0:k_R-k_L
    for ii= 0:K-1
        temp = x(ii+1)*cos(2*pi*(k+k_L)*ii/N);
        a(k+1)=a(k+1) + 2/N*temp;
        b(k+1)=b(k+1) +2/N*x(ii+1)*sin(2*pi*(k+k_L)*ii/N);
        axx(k+1)=axx(k+1) + 2/N*acc_slice(ii+1)*cos(2*pi*(k+k_L)*ii/N);
        bxx(k+1)=bxx(k+1) +2/N*acc_slice(ii+1)*sin(2*pi*(k+k_L)*ii/N);
    end
%     c(k+1)=sqrt(a(k+1)^2+b(k+1)^2);
%     cxx(k+1)=sqrt(axx(k+1)^2+bxx(k+1)^2);
    c(k+1)=(a(k+1)^2+b(k+1)^2);
    cxx(k+1)=(axx(k+1)^2+bxx(k+1)^2);

end

f=(k_L:k_R)*df;

c = c/max(c);

cxx = cxx/max(cxx);

if ppgstart==3901
a_c = c;
a_cxx = cxx;
end

c = c-cxx;

[pks,locs] = findpeaks(c);

range = 30;

if ppgstart ==1
    ind = 30;
else
    ind = mean(ind_list(max(1, end-20):end));
% ind = ind_list(end);
end

[~,index_peak] = max(pks(abs(locs-ind)<range));
temp_locs = locs((abs(locs-ind)<range));
if ~isempty(index_peak) 
ind = temp_locs(index_peak);
end


%%
ind_list = [ind_list floor(ind)];

bpm = [bpm f(ind_list(end))*60];

z_ppg1(i) = f(ind_list(end))*60;
if ppgstart==3901
a_hr = z_ppg1(i);
a_i = i;
end
% if i >700
% subplot(3,1,1);
% plot(x);
% subplot(3,1,2);
% plot(c_ori);
% hold on
% plot(cxx);
% hold off
% subplot(3,1,3);
% plot(c);
% hold on 
% plot([ind, ind], [c(ind), c(ind)], 'ro');
% true_hr = floor(z_polarHR_data(i))-k_L+1;
% plot([true_hr, true_hr], [c(true_hr), c(true_hr)], 'go');
% hold off
% pause();
% end

i = i+1;
ppgstart = ppgstart+25;
ppgend = ppgend+25;
end

figure;
plot(z_ppg1);
title('估计HR');
%%
new_z_ppg2 = [];

for i = 1:length(z_ppg1)
%     new_z_ppg2(i) = mean(z_ppg1(max(1, i-10):min(length(z_ppg1), i+10)));
    new_z_ppg2(i) = mean(z_ppg1(max(1, i-20):i));
end

figure;
plot(new_z_ppg2);
title('均值HR');
z_ppg1 = new_z_ppg2;

%%
z_ppg2 = resample(z_ppg1, length(polarHR_data), length(z_ppg1));

figure;
plot(polarHR_data,'LineWidth',1);
hold on 
plot(z_ppg2,'LineWidth',1,'LineStyle','-.');

legend('polar HR','ppg HR');
mean_dif = mean(abs(polarHR_data-z_ppg2));
% title(num2str(mean_dif));
end
