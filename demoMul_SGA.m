clear; close all;

%% load weizmann data
acts = {'bend', 'run', 'pjump', 'walk', 'jack', 'wave1', 'side', 'jump', 'wave2', 'skip'};
num_acts = length(acts);
X = {};
for i=1:num_acts
    data_file = strcat('./data/weizmann/',acts{i},'.mat');
    load(data_file);
    X{i} = Xs;
end

%% concatenate action videos w.r.t subject id
subIdx1 = 1; subIdx2 = 2; subIdx3 = 3; xDim = 123; 
X1 = []; X2 = []; X3 = []; y1 = []; y2 = []; y3 = [];
for i=1:num_acts
    % for subject 1
    X1 = [X1 X{i}{subIdx1}(1:xDim,:)];
    [~,nF] = size(X{i}{subIdx1}); 
    y1 = [y1 i*ones(1,nF)];
    % for subject 2
    X2 = [X2 X{i}{subIdx2}(1:xDim,:)];
    [~,nF] = size(X{i}{subIdx2}); 
    y2 = [y2 i*ones(1,nF)];
    % for subject 3
    X3 = [X3 X{i}{subIdx3}(1:xDim,:)];
    [~,nF] = size(X{i}{subIdx3}); 
    y3 = [y3 i*ones(1,nF)];
end
Xs = {X1,X2,X3}; 
ys = [y1,y2,y3];
clear data_file i nF X X0s X1 X2 X3 XTs y1 y2 y3;

%% multi kcsr parameters
params.m = 10;
params.alpha = 10;
params.s = 7;
params.kn = @knGauss;
params.lambda = 1e-5;
params.batchsize = 450;
params.maxepoch = 155;
params.eta_0 = 0.01;
params.decay = 1;
params.momentum = 0.99;
params.seed = 6000;

%% learn the model
t0 = tic;
model = KCSR_balanced_Multi_SGAm(Xs,params);
t1 = toc(t0);
fprintf('Algorithm has stopped. Execution time is %.10f \n',t1);

%% compare with the ground truth
res = bestMap(ys,round(model.tau));
MIhat = MutualInfo(ys,res); 
fprintf('Normalized Mutual Information %.5f \n',MIhat);

%% plot
figure(1); act_ticks = {'bend', 'run', 'pjump', 'walk', 'jack', 'wave1', 'side', 'jump', 'wave2', 'skip', 'bend', 'run', 'pjump', 'walk', 'jack', 'wave1', 'side', 'jump', 'wave2', 'skip', 'bend', 'run', 'pjump', 'walk', 'jack', 'wave1', 'side', 'jump', 'wave2', 'skip'};
% ground truth
subplot(3,1,1);
imagesc(ys);
title('ground truth');
anchors = [1 find(diff(ys) == 1 | diff(ys) == -9) numel(ys)];
ticks = anchors(1:end-1) + round(diff(anchors)/2);
set(gca,'xtick',ticks);
set(gca,'xticklabel',act_ticks);
set(gca,'TickLength',[0 0]);
set(gca,'ytick',[]);
set(gca,'yticklabel',[]);
set (gca,'FontSize',20);
% initial segmentation
subplot(3,1,2);
y_init=round(model.tau_init);
imagesc(y_init);
title('initial segmentation');
anchors = [1 find(diff(y_init) == 1 | diff(y_init) == -9) numel(y_init)];
ticks = anchors(1:end-1) + round(diff(anchors)/2);
set(gca,'xtick',ticks);
set(gca,'xticklabel',act_ticks);
set(gca,'TickLength',[0 0]);
set(gca,'ytick',[]);
set(gca,'yticklabel',[]);
set (gca,'FontSize',20);
% final segmentation
subplot(3,1,3);
y_final=round(model.tau);
imagesc(y_final);
title('final segmentation');
anchors = [1 find(diff(y_final) == 1 | diff(y_final) == -9) numel(y_final)];
ticks = anchors(1:end-1) + round(diff(anchors)/2);
set(gca,'xtick',ticks);
set(gca,'xticklabel',act_ticks);
set(gca,'TickLength',[0 0]);
set(gca,'ytick',[]);
set(gca,'yticklabel',[]);
set (gca,'FontSize',20)
% color map
map = [0 0.4470 0.7410
    0.8500 0.3250 0.0980
    0.9290 0.6940 0.1250
    0.4940 0.1840 0.5560
    0.4660 0.6740 0.1880
    0.3010 0.7450 0.9330
    0.6350 0.0780 0.1840
    0.25, 0.25, 0.25
    0.75, 0, 0.75
    0, 0.5, 0];
colormap(map);