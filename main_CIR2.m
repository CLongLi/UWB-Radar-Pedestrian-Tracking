function main_CIR2
clear;clc
load('Bg_CIR_VAR.mat');
load('Dyn_CIR_VAR.mat');
load('AnchorPos.mat')

%% prediction labels
diff_ToF01 = abs(Dyn_real_ToF01-ToF_TRx01);
diff_ToF02 = abs(Dyn_real_ToF02-ToF_TRx02);
diff_ToF04 = abs(Dyn_real_ToF04-ToF_TRx04);
diff_ToF12 = abs(Dyn_real_ToF12-ToF_TRx12);
diff_ToF14 = abs(Dyn_real_ToF14-ToF_TRx14);
diff_ToF24 = abs(Dyn_real_ToF24-ToF_TRx24);

label01 = zeros(numel(diff_ToF01),1);
label02 = zeros(numel(diff_ToF02),1);
label04 = zeros(numel(diff_ToF04),1);
label12 = zeros(numel(diff_ToF12),1);
label14 = zeros(numel(diff_ToF14),1);
label24 = zeros(numel(diff_ToF24),1);
for i = 1:numel(diff_ToF01)
    [~,label01(i,1)] = min(abs(diff_ToF01(i)-re_SampTime));
end
for i = 1:numel(diff_ToF02)
    [~,label02(i,1)] = min(abs(diff_ToF02(i)-re_SampTime));
end
for i = 1:numel(diff_ToF04)
    [~,label04(i,1)] = min(abs(diff_ToF04(i)-re_SampTime));
end
for i = 1:numel(diff_ToF12)
    [~,label12(i,1)] = min(abs(diff_ToF12(i)-re_SampTime));
end
for i = 1:numel(diff_ToF14)
    [~,label14(i,1)] = min(abs(diff_ToF14(i)-re_SampTime));
end
for i = 1:numel(diff_ToF24)
    [~,label24(i,1)] = min(abs(diff_ToF24(i)-re_SampTime));
end

%% test
tstSet01 = find(Dyn_re_tUWB01>47.6 & Dyn_re_tUWB01<73.8);
tstSet02 = find(Dyn_re_tUWB02>47.6 & Dyn_re_tUWB02<73.8);
tstSet04 = find(Dyn_re_tUWB04>47.6 & Dyn_re_tUWB04<73.8);
tstSet12 = find(Dyn_re_tUWB12>47.6 & Dyn_re_tUWB12<73.8);
tstSet14 = find(Dyn_re_tUWB14>47.6 & Dyn_re_tUWB14<73.8);
tstSet24 = find(Dyn_re_tUWB24>47.6 & Dyn_re_tUWB24<73.8);

num_tst01 = numel(tstSet01);
num_tst02 = numel(tstSet02);
num_tst04 = numel(tstSet04);
num_tst12 = numel(tstSet12);
num_tst14 = numel(tstSet14);
num_tst24 = numel(tstSet24);
%% train
trainSet01 = find(Dyn_re_tUWB01>73.8 | Dyn_re_tUWB01<47.6);
trainSet02 = find(Dyn_re_tUWB02>73.8 | Dyn_re_tUWB02<47.6);
trainSet04 = find(Dyn_re_tUWB04>73.8 | Dyn_re_tUWB04<47.6);
trainSet12 = find(Dyn_re_tUWB12>73.8 | Dyn_re_tUWB12<47.6);
trainSet14 = find(Dyn_re_tUWB14>73.8 | Dyn_re_tUWB14<47.6);
trainSet24 = find(Dyn_re_tUWB24>73.8 | Dyn_re_tUWB24<47.6);

trainIDX01 = numel(trainSet01);
trainIDX02 = numel(trainSet02);
trainIDX04 = numel(trainSet04);
trainIDX12 = numel(trainSet12);
trainIDX14 = numel(trainSet14);
trainIDX24 = numel(trainSet24);
for  i = 1:trainIDX01
    X_train_tmp01(:,1,1,i) = mat2gray(abs(Dyn_re_CIR01(trainSet01(i,1),:))');
    X_train_tmp01(:,2,1,i) = mat2gray(abs(Bg_re_CIR01)');
end
for  i = 1:trainIDX02
    X_train_tmp02(:,1,1,i) = mat2gray(abs(Dyn_re_CIR02(trainSet02(i,1),:))');
    X_train_tmp02(:,2,1,i) = mat2gray(abs(Bg_re_CIR02)');
end
for  i = 1:trainIDX04
    X_train_tmp04(:,1,1,i) = mat2gray(abs(Dyn_re_CIR04(trainSet04(i,1),:))');
    X_train_tmp04(:,2,1,i) = mat2gray(abs(Bg_re_CIR04)');
end
for  i = 1:trainIDX12
    X_train_tmp12(:,1,1,i) = mat2gray(abs(Dyn_re_CIR12(trainSet12(i,1),:))');
    X_train_tmp12(:,2,1,i) = mat2gray(abs(Bg_re_CIR12)');
end
for  i = 1:trainIDX14
    X_train_tmp14(:,1,1,i) = mat2gray(abs(Dyn_re_CIR14(trainSet14(i,1),:))');
    X_train_tmp14(:,2,1,i) = mat2gray(abs(Bg_re_CIR14)');
end
for  i = 1:trainIDX24
    X_train_tmp24(:,1,1,i) = mat2gray(abs(Dyn_re_CIR24(trainSet24(i,1),:))');
    X_train_tmp24(:,2,1,i) = mat2gray(abs(Bg_re_CIR24)');
end
X_train_tmp = cat(4,X_train_tmp01,X_train_tmp02,X_train_tmp04,X_train_tmp12,X_train_tmp14,X_train_tmp24);
Y_train_tmp = [label01(trainSet01',:);label02(trainSet02',:);label04(trainSet04',:);label12(trainSet12',:);label14(trainSet14',:);label24(trainSet24',:)];
% shuffle
trainIDX = trainIDX01+trainIDX02+trainIDX04+trainIDX12+trainIDX14+trainIDX24;
RDidx = randperm(trainIDX);
trainIDX_tmp = floor(trainIDX*0.85);
X_train = X_train_tmp(:,:,:,RDidx(1:trainIDX_tmp));
Y_train = Y_train_tmp(RDidx(1:trainIDX_tmp),:);
X_val = X_train_tmp(:,:,:,RDidx(trainIDX_tmp+1:end));
Y_val = Y_train_tmp(RDidx(trainIDX_tmp+1:end),:);

if ~isfile(['ConvNet_CIR2.mat'])
    ConvNet_CIR2 = CIR_CNN_CIRVar_Tst(X_train,Y_train,X_val,Y_val,"CIR");
    save ConvNet_CIR2.mat ConvNet_CIR2
else
    load(['ConvNet_CIR2.mat']);
end

%% test
for i = 1:num_tst01
    X_test01(:,1,1,i) = mat2gray(abs(Dyn_re_CIR01(tstSet01(i,1),:))');
    X_test01(:,2,1,i) = mat2gray(abs(Bg_re_CIR01)');
end
for i = 1:num_tst02
    X_test02(:,1,1,i) = mat2gray(abs(Dyn_re_CIR02(tstSet02(i,1),:))');
    X_test02(:,2,1,i) = mat2gray(abs(Bg_re_CIR02)');
end
for i = 1:num_tst04
    X_test04(:,1,1,i) = mat2gray(abs(Dyn_re_CIR04(tstSet04(i,1),:))');
    X_test04(:,2,1,i) = mat2gray(abs(Bg_re_CIR04)');
end
for i = 1:num_tst12
    X_test12(:,1,1,i) = mat2gray(abs(Dyn_re_CIR12(tstSet12(i,1),:))');
    X_test12(:,2,1,i) = mat2gray(abs(Bg_re_CIR12)');
end
for i = 1:num_tst14
    X_test14(:,1,1,i) = mat2gray(abs(Dyn_re_CIR14(tstSet14(i,1),:))');
    X_test14(:,2,1,i) = mat2gray(abs(Bg_re_CIR14)');
end
for i = 1:num_tst24
    X_test24(:,1,1,i) = mat2gray(abs(Dyn_re_CIR24(tstSet24(i,1),:))');
    X_test24(:,2,1,i) = mat2gray(abs(Bg_re_CIR24)');
end
Y_test01 = label01(tstSet01',:);
Y_test02 = label02(tstSet02',:);
Y_test04 = label04(tstSet04',:);
Y_test12 = label12(tstSet12',:);
Y_test14 = label14(tstSet14',:);
Y_test24 = label24(tstSet24',:);
%%
tic
Y_pred01 = predict(ConvNet_CIR2,X_test01);
Y_pred02 = predict(ConvNet_CIR2,X_test02);
Y_pred04 = predict(ConvNet_CIR2,X_test04);
Y_pred12 = predict(ConvNet_CIR2,X_test12);
Y_pred14 = predict(ConvNet_CIR2,X_test14);
Y_pred24 = predict(ConvNet_CIR2,X_test24);
t_CNN = toc;
t_CNN = t_CNN/(num_tst01+num_tst02+num_tst04+num_tst12+num_tst14+num_tst24)*6*1000; % ms

SampDiff = abs(re_SampTime(2)-re_SampTime(1));
for i = 1:num_tst01
    ToF_est01(i) = SampDiff*Y_pred01(i,1)+re_SampTime(1)+ToF_TRx01;
end
for i = 1:num_tst02
    ToF_est02(i) = SampDiff*Y_pred02(i,1)+re_SampTime(1)+ToF_TRx02;
end
for i = 1:num_tst04
    ToF_est04(i) = SampDiff*Y_pred04(i,1)+re_SampTime(1)+ToF_TRx04;
end
for i = 1:num_tst12
    ToF_est12(i) = SampDiff*Y_pred12(i,1)+re_SampTime(1)+ToF_TRx12;
end
for i = 1:num_tst14
    ToF_est14(i) = SampDiff*Y_pred14(i,1)+re_SampTime(1)+ToF_TRx14;
end
for i = 1:num_tst24
    ToF_est24(i) = SampDiff*Y_pred24(i,1)+re_SampTime(1)+ToF_TRx24;
end
%% ToF error distribution
Y_pred_val_tmp = predict(ConvNet_CIR2,X_val);
DiffY_val = SampDiff*(Y_val-Y_pred_val_tmp);
DiffY_val = double(DiffY_val);
pd = fitdist(DiffY_val,'tLocationScale');

Time_pair01 = Dyn_re_tUWB01(tstSet01')';
Time_pair02 = Dyn_re_tUWB02(tstSet02')';
Time_pair04 = Dyn_re_tUWB04(tstSet04')';
Time_pair12 = Dyn_re_tUWB12(tstSet12')';
Time_pair14 = Dyn_re_tUWB14(tstSet14')';
Time_pair24 = Dyn_re_tUWB24(tstSet24')';

tst_MU01 = Dyn_re_MU01(tstSet01',:);
tst_MU02 = Dyn_re_MU02(tstSet02',:);
tst_MU04 = Dyn_re_MU04(tstSet04',:);
tst_MU12 = Dyn_re_MU12(tstSet12',:);
tst_MU14 = Dyn_re_MU14(tstSet14',:);
tst_MU24 = Dyn_re_MU24(tstSet24',:);

resamp4tracking

%% four nodes, input format: ToF_est10,ToF_est12,ToF_est14,ToF_est20,ToF_est24,ToF_est40
ParticleFilter4Nodes(ToF_est10_new,ToF_est12_new,ToF_est14_new,ToF_est20_new,ToF_est24_new,ToF_est40_new,Sel_MU_new,AnchorPos,TIME_reshape,pd);

end
