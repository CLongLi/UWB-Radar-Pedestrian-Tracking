function ParticleFilter4Nodes(ToF_est10,ToF_est12,ToF_est14,ToF_est20,ToF_est24,ToF_est40,Pos_MU,Pos_Nodes,tUWB,pd)
Pos_Node0 = Pos_Nodes(1,:)';
Pos_Node1 = Pos_Nodes(2,:)';
Pos_Node2 = Pos_Nodes(3,:)';
Pos_Node4 = Pos_Nodes(4,:)';

PosXY_Node0 = Pos_Node0(1:2,:);
PosXY_Node1 = Pos_Node1(1:2,:);
PosXY_Node2 = Pos_Node2(1:2,:);
PosXY_Node4 = Pos_Node4(1:2,:);
%% t location-scale distribution: PDF paras
sigma = pd.sigma;
mu = pd.mu;
nu = pd.nu;
%% initialize the variables  
light_speed = 3e8;
v_std_noise = 5; % x,y
Np = 200; % particle number 
x_P = zeros(4,Np);  % particles
x_P_update = zeros(2,Np);
ToF_update10 = zeros(1,Np);
ToF_update12 = zeros(1,Np);
ToF_update14 = zeros(1,Np);
ToF_update20 = zeros(1,Np);
ToF_update24 = zeros(1,Np);
ToF_update40 = zeros(1,Np);
P_w = zeros(1,Np);
 
x_P(1,:) = -5+8*rand(1,Np);
x_P(2,:) = -5+6*rand(1,Np);
x_P(3,:) = zeros(1,Np);
x_P(4,:) = zeros(1,Np);

x_est = zeros(numel(tUWB)-1,2);

tic
for t = 1:numel(tUWB)-1
    delta_t = tUWB(t+1)-tUWB(t);
    for i = 1:Np
        % p(x(k)|x(k-1))   
        x_P_update(1,i) = x_P(1,i) + delta_t*v_std_noise*randn;  
        x_P_update(2,i) = x_P(2,i) + delta_t*v_std_noise*randn;
        % 
        ToF_update10(i) = (norm(PosXY_Node1-x_P_update(1:2,i))+norm(PosXY_Node0-x_P_update(1:2,i)))/light_speed*1e9;
        ToF_update12(i) = (norm(PosXY_Node1-x_P_update(1:2,i))+norm(PosXY_Node2-x_P_update(1:2,i)))/light_speed*1e9;
        ToF_update14(i) = (norm(PosXY_Node1-x_P_update(1:2,i))+norm(PosXY_Node4-x_P_update(1:2,i)))/light_speed*1e9;
        ToF_update20(i) = (norm(PosXY_Node2-x_P_update(1:2,i))+norm(PosXY_Node0-x_P_update(1:2,i)))/light_speed*1e9;
        ToF_update24(i) = (norm(PosXY_Node2-x_P_update(1:2,i))+norm(PosXY_Node4-x_P_update(1:2,i)))/light_speed*1e9; 
        ToF_update40(i) = (norm(PosXY_Node4-x_P_update(1:2,i))+norm(PosXY_Node0-x_P_update(1:2,i)))/light_speed*1e9; 
        % update weight  
        % t location scale
        P_w0 = tLocationScale_pdf(ToF_est10(t)-ToF_update10(i),sigma,mu,nu);
        P_w1 = tLocationScale_pdf(ToF_est12(t)-ToF_update12(i),sigma,mu,nu);
        P_w2 = tLocationScale_pdf(ToF_est14(t)-ToF_update14(i),sigma,mu,nu);
        P_w3 = tLocationScale_pdf(ToF_est20(t)-ToF_update20(i),sigma,mu,nu);
        P_w4 = tLocationScale_pdf(ToF_est24(t)-ToF_update24(i),sigma,mu,nu);
        P_w5 = tLocationScale_pdf(ToF_est40(t)-ToF_update40(i),sigma,mu,nu);
        
        P_w(i) = P_w0*P_w1*P_w2*P_w3*P_w4*P_w5; 
    end  
    % normalization 
    P_w = P_w./sum(P_w);  
    %% Resampling    
    [~, idx_tmp] = histc(rand(Np,1), [0 cumsum(P_w)]); 
    x_P = x_P_update(:,idx_tmp);
    x_est(t,:) = mean(x_P(1:2,:)',1);        
end  
t_PF = toc;
t_PF = t_PF/(numel(tUWB)-1)*1000;

figure;hold on;

plot(Pos_Nodes(:,1),Pos_Nodes(:,2),'b^');
plot(Pos_MU(:,1),Pos_MU(:,2),'r.');
plot(x_est(:,1),x_est(:,2),'k.');
xlabel('$x$ (m)'); ylabel('$y$ (m)');
xlim([-5, 4.1])
ylim([-5, 1])
legend('Ground truth','Tracking results')
L = legend;L.ItemTokenSize(1) = 15;

disERR = sqrt((x_est(:,1)-Pos_MU(1:end-1,1)).^2+(x_est(:,2)-Pos_MU(1:end-1,2)).^2);

figure;
cdfdraw(disERR,'color','black','LineStyle','-.','Marker','none')
xlabel('Tracking errors (m)')

end

function y = tLocationScale_pdf(x,sigma,mu,nu)
y1 = gamma((nu+1)/2)/(sigma*sqrt(nu*pi)*gamma(0.5*nu));
y2 = ((nu+((x-mu)/sigma).^2)/nu)^(-0.5*(nu+1));
y = y1*y2;
y(y<1e-20) = 1e-20; % avoid zero-probability in Matlab
end

