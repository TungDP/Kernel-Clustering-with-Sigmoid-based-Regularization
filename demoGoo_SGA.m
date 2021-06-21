clear; close all;
warning('off');

addpath('./mfcc');

%% load digist sound
digit = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'};
audio_clean = [];
for i=1:numel(digit)
    wav_file = strcat('./data/google/',digit{i},'.wav');
    [signal, fs] = audioread(wav_file);
    audio_clean = [audio_clean; signal];
end
n = numel(audio_clean);

%% load noise
wav_file = './data/google/white_noise.wav';
[noise, fs] = audioread(wav_file);

%% add noise to audio signal
audio_noisy = audio_clean + 0.15 * noise(1:n);

%% signal processing variables
Tw = 25;                % frame duration (ms)
Ts = 10;                % frame shift (ms)
alpha = 0.97;           % preemphasis coefficient
M = 20;                 % number of filterbank channels 
C = 12;                 % number of cepstral coefficients
L = 22;                 % cepstral sine lifter parameter
LF = 300;               % lower frequency limit (Hz)
HF = 3700;              % upper frequency limit (Hz)

%% compute Mel-frequency cepstral coefficients and log FBEs of clean audio
[ MFCCs_clean, FBEs_clean, frames ] = ...
            mfcc( audio_clean, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
logFBEs_clean = 20*log10(FBEs_clean);                       % compute log FBEs for plotting
logFBEs_floor_clean = max(logFBEs_clean(:))-50;             % get logFBE floor 50 dB below max
logFBEs_clean(logFBEs_clean<logFBEs_floor_clean) = logFBEs_floor_clean; % limit logFBE dynamic range
%X = logFBEs_clean;

%% compute Mel-frequency cepstral coefficients and log FBEs of noisy audio
[ MFCCs_noisy, FBEs_noisy, frames ] = ...
            mfcc( audio_noisy, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
logFBEs_noisy = 20*log10(FBEs_noisy);                       % compute log FBEs for plotting
logFBEs_floor_noisy = max(logFBEs_noisy(:))-50;             % get logFBE floor 50 dB below max
logFBEs_noisy(logFBEs_noisy<logFBEs_floor_noisy) = logFBEs_floor_noisy; % limit logFBE dynamic range
%X = logFBEs_noisy; 
X = MFCCs_noisy;

%% manual label on the logFBEs_clean
change_points = [32,69,131,168,209,264,322,352,423,476,536,568,631,686,725,759,814,854];
stops = [0 change_points size(X,2)]; gnd = [];
for i=1:numel(stops)-1
    gnd = [gnd i*ones(1,stops(i+1) - stops(i))];
end    

%% kssr parameters
params.m = 19;
params.alpha = 10;
params.s = 8.5; % 8.5 for MFCCs_noisy
params.kn = @knGauss;
params.lambda = 1e-2;
params.batchsize = 500; % 300
params.maxepoch = 100; % 100
params.eta_0 = 0.01; % 0.01
params.decay = 1;
params.momentum = 0.99;
params.seed = 600;

%% learn the model
t0 = tic;
model = KCSR_balanced_SGAm(X,params);
t1 = toc(t0);
fprintf('Algorithm has stopped. Execution time is %.10f \n',t1);

%% compare with the ground truth
res = bestMap(gnd,round(model.tau));
MIhat = MutualInfo(gnd,res); 
fprintf('Normalized Mutual Information %.5f \n',MIhat);

%% plot
time = 1:1:n; time_frames = 1:1:size(MFCCs_clean,2); beta = model.beta;

figure(1);

% plot clean audio
subplot(4,1,1);
plot(time,audio_clean,'Color',[0.25, 0.25, 0.25]);
xlim([ min(time) max(time)]);
xlabel('Time (samples)');
yticks([-1 0 1]);
yticklabels({'-1','0','1'});
ylabel('Amplitude'); 
title('Clean audio');
set(gca,'FontSize',20);

% plot noisy audio
subplot(4,1,2);
plot(time,audio_noisy,'Color',[0.25, 0.25, 0.25]);
xlim([min(time) max(time)]);
xlabel('Time (samples)');
yticks([-1 0 1]);
yticklabels({'-1','0','1'});
ylabel('Amplitude'); 
title('Noisy audio');
set(gca,'FontSize',20);

% plot clean logFBEs
subplot(4,1,3);
imagesc(time_frames,[1:M],logFBEs_clean);
y1=get(gca,'ylim'); hold on;
for i=1:numel(beta) 
    plot([change_points(i) change_points(i)], y1, 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410]); hold on;
end
axis('xy');
xlim([ min(time_frames) max(time_frames)]);
xticks([1 100 200 300 400 500 600 700 800 898]);
xticklabels({'1', '100', '200', '300', '400', '500', '600', '700', '800', '898'});
xlabel('Time (frames)'); 
ylabel('Channel index'); 
title('Clean log (mel) filterbank energies');
set(gca,'FontSize',20);

% plot MFCCs
subplot(4,1,4);
imagesc(time_frames,[1:C],MFCCs_noisy(2:C,:)); % HTK's TARGETKIND: MFCC
%imagesc( time_frames, [1:C+1], MFCCs );   % HTK's TARGETKIND: MFCC_0
y1=get(gca,'ylim'); hold on;
for i=1:numel(beta) 
    plot([beta(i) beta(i)], y1, 'LineWidth', 2.5, 'Color', [0 1 0.7586]); hold on;
end
axis('xy');
xlim([ min(time_frames) max(time_frames)]);
xticks([1 100 200 300 400 500 600 700 800 898]);
xticklabels({'1', '100', '200', '300', '400', '500', '600', '700', '800', '898'});
xlabel('Time (frames)'); 
ylabel('Cepstrum index');
title('Mel frequency cepstrum');
set(gca,'FontSize',20);

% Set color map to grayscale
colormap(1-colormap('gray'));