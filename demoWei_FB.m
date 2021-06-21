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

%% concatenate data
subIdx = 1; xDim = 123; 
Xpca = [];  y = [];
for i=1:num_acts   
    Xpca = [Xpca X{i}{subIdx}(1:xDim,:)];
    [~,nF] = size(X{i}{subIdx}); 
    y = [y i*ones(1,nF)];
end

%% kssr parameters
params.m = 10;
params.alpha = 10;
params.s = 7;
params.kn = @knGauss;
params.lambda = 1e-3;
params.eta_0 = 0.15;
params.tolerance = 1e-8;
params.seed = 666;

%% learn the model
t0 = tic;
model = KCSR_balanced_FB(Xpca,params);
t1 = toc(t0);
fprintf('Algorithm has stopped. Execution time is %.10f \n',t1);

%% compare with the ground truth
res = bestMap(y,round(model.tau));
MIhat = MutualInfo(y,res); 
fprintf('Normalized Mutual Information %.5f \n',MIhat);

%% plot
figure(1);
% ground truth
subplot(3,1,1);
imagesc(y);
title('ground truth');
anchors = [1 find(diff(y) == 1) numel(y)];
ticks = anchors(1:end-1) + round(diff(anchors)/2);
set(gca,'xtick',ticks);
set(gca,'xticklabel',acts);
set(gca,'TickLength',[0 0]);
set(gca,'ytick',[]);
set(gca,'yticklabel',[]);
set (gca,'FontSize',20);
% initial segmentation
subplot(3,1,2);
y_init=round(model.tau_init);
imagesc(y_init);
title('initial segmentation');
anchors = [1 find(diff(y_init) == 1) numel(y_init)];
ticks = anchors(1:end-1) + round(diff(anchors)/2);
set(gca,'xtick',ticks);
set(gca,'xticklabel',acts);
set(gca,'TickLength',[0 0]);
set(gca,'ytick',[]);
set(gca,'yticklabel',[]);
set (gca,'FontSize',20);
% final segmentation
subplot(3,1,3);
y_final=round(model.tau);
imagesc(y_final);
title('final segmentation');
anchors = [1 find(diff(y_final) == 1) numel(y_final)];
ticks = anchors(1:end-1) + round(diff(anchors)/2);
set(gca,'xtick',ticks);
set(gca,'xticklabel',acts);
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