Time_pair_tmp = [Time_pair01 Time_pair02 Time_pair04 Time_pair12 Time_pair14 Time_pair24];
[Time_pair_tmp,idx_tmp] = sort(Time_pair_tmp+(1e-8)*rand(size(Time_pair_tmp)));

tst_MU_tmp = [tst_MU01;tst_MU02;tst_MU04;tst_MU12;tst_MU14;tst_MU24];
tst_MU_tmp = tst_MU_tmp(idx_tmp,:);

numTime = 10000;
t_START = Time_pair_tmp(1);
t_END = Time_pair_tmp(end);
TIME_reshape = t_START:(t_END-t_START)/numTime:t_END;

ToF_est10_new = interp1(sort(Time_pair01+(1e-8)*rand(1,numel(ToF_est01))),ToF_est01,TIME_reshape,'linear','extrap');
ToF_est12_new = interp1(sort(Time_pair12+(1e-8)*rand(1,numel(ToF_est12))),ToF_est12,TIME_reshape,'linear','extrap');
ToF_est14_new = interp1(sort(Time_pair14+(1e-8)*rand(1,numel(ToF_est14))),ToF_est14,TIME_reshape,'linear','extrap');
ToF_est20_new = interp1(sort(Time_pair02+(1e-8)*rand(1,numel(ToF_est02))),ToF_est02,TIME_reshape,'linear','extrap');
ToF_est24_new = interp1(sort(Time_pair24+(1e-8)*rand(1,numel(ToF_est24))),ToF_est24,TIME_reshape,'linear','extrap');
ToF_est40_new = interp1(sort(Time_pair04+(1e-8)*rand(1,numel(ToF_est04))),ToF_est04,TIME_reshape,'linear','extrap');

Sel_MU_new(:,1) = interp1(Time_pair_tmp,tst_MU_tmp(:,1),TIME_reshape);
Sel_MU_new(:,2) = interp1(Time_pair_tmp,tst_MU_tmp(:,2),TIME_reshape);
Sel_MU_new(:,3) = interp1(Time_pair_tmp,tst_MU_tmp(:,3),TIME_reshape);






