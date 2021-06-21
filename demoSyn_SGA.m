clear; close all;

%% load synthetic data
load ./data/syn/four_circles.mat

%% kssr parameters
params.m = 4;
params.alpha = 10;
params.s = 0.05;
params.kn = @knGauss;
params.lambda = 1e-6;
params.batchsize = 250;
params.maxepoch = 100;
params.eta_0 = 0.01; %0.15; for grad params.batchsize % 0.01; for standard
params.decay = 1;
params.momentum = 0.99;
params.seed = 6000;

%% learn the model
t0 = tic;
model = KCSR_balanced_SGAm(X,params);
t1 = toc(t0);
fprintf('Algorithm has stopped. Execution time is %.10f \n',t1);

%% evaluate the algorithm
res = bestMap(label,round(model.tau));
MIhat = MutualInfo(label,res); 
fprintf('Normalized Mutual Information %.5f \n',MIhat);

%% plot
colors = [[0 0.4470 0.7410]; [0.6350 0.0780 0.1840]; [0.4660 0.6740 0.1880]; [0.4940 0.1840 0.5560]];
time = (1:1:numel(label)); s = 50 * ones(numel(label),1);
figure(1)
% input data
subplot(131)
c = [0 0.4470 0.7410];
scatter3(time,X(1,:),X(2,:),s,c);
title('(a) input data')
% initial segments
subplot(132)
y = round(model.tau_init);
c = colors(y,:);
scatter3(time,X(1,:),X(2,:),s,c);
title('(b) initial segmentation')
% final segments
subplot(133)
y = round(model.tau);
c = colors(y,:);  
scatter3(time,X(1,:),X(2,:),s,c);
title('(c) final segmentation')
figure(2)
plot(model.objs,'LineWidth',2,'Color',[0 0.4470 0.7410]);