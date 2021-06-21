function model = KCSR_balanced_SGAo(X,params,p)
%Inputs
%   X       :  data sequence (d x n)
%   params  :
%       params.m        :  number of segments
%       params.alpha    :  shared paramter of the mixture of sigmoids
%       params.s        :  sigma for kernel function
%       params.kn       :  kernel function
%       params.lambda   :  paramter control balance regularization term
%       params.batchsize:  minibatch size
%       params.maxepoch :  number of time pass over the sequence
%       params.eta_0    :  initial step size
%       params.decay    :  the geometric rate in which the learning rate decays.
%       params.momentum :  momentum parameter for SGA, with value in [0 1).
%       p               :  ground truth change points
%Outputs
%   model   :
%       model.gamma_init
%       model.beta_init
%       model.tau_int
%       model.G_init
%
%       model.gamma :  hyper paramters for beta (1 x m)
%       model.beta  :  paramters of the mixture of sigmoids (1 x (m-1))
%       model.tau   :  sample-segment indicator (1 x n)
%       model.G     :  sample-segment indicator matrix (m x n)
%       model.ress  :  residuals between computed beta and ground truth change points p
%       model.times :  time spent for each epoch
%
%       model.params:  model paramters
%           model.params.m        :  number of segments
%           model.params.alpha    :  shared paramter of the mixture of sigmoids
%           model.params.s        :  sigma for kernel function
%           model.params.kn       :  kernel function
%           model.params.lambda   :  paramter control balance regularization term
%           model.params.batchsize:  minibatch size
%           model.params.maxepoch :  number of time pass over the sequence
%           model.params.eta_0    :  initial step size
%           model.params.decay    :  the geometric rate in which the learning rate decays.
%           model.params.momentum :  momentum parameter for SGD, with value in [0 1).

%% parameter
[~,n] = size(X);
params.n = n;

%% initialize gamma
gamma = init_g(params.m,params.seed);
fprintf('initial gamma = %s\n',num2str(gamma));

%% save initial model
[G,tau,beta] = forwardpass(gamma,params);
res_opt = sum(abs(beta - p)) / (params.m - 1);

model.gamma_init = gamma;
model.beta_init = beta;
model.tau_init = tau;
model.G_init = G;

model.gamma = gamma;
model.beta = beta;
model.tau = tau;
model.G = G;
model.ress = [res_opt];
model.times = [];

%% run stochastic gradient ascent
fprintf('initial beta = %s\n',num2str(beta));
fprintf('initial residual value = %.5f\n',model.ress(end));
fprintf('start stochastic gradient ascent ...\n');

numbatches = ceil(n/params.batchsize); its=0;
while its < params.maxepoch
    
    eta_init = params.eta_0 * params.decay ^ its; % Reduce learning rate.
    params.eta_init = eta_init;      
    t0=tic;
    rp = randperm(n); % Shuffle the data set.
    for i=1:numbatches
        % extract minibatch
        start = (i-1)*params.batchsize+1;
        stop  = min(i*params.batchsize,n);
        idxs  = sort([rp(start:stop),rp(1:max(0,i*params.batchsize-n))]);
        % compute kernel matrix for the minibatch
        K_mini = params.kn(X(:,idxs),X(:,idxs),params.s);
        % estimate stochastic gradient
        grad = backwardpass(idxs,K_mini,G(:,idxs),tau(idxs),beta,gamma,params);
        % estimate step size
        eta = backtracking(idxs,K_mini,gamma,grad,params);   
        % update gamma
        gamma = gamma + eta * grad;
    end
    
    %% record the time spent for each epoch.
    its=its+1; model.times=[model.times, toc(t0)]; params = rmfield(params,'eta_init');
    
    %% check residual
    [G,tau,beta] = forwardpass(gamma,params);
    res = sum(abs(beta - p)) / (params.m - 1);
    model.ress = [model.ress res]; % save objective value
    if res < res_opt
        % save model if objective is improved
        fprintf('epoch %d: getting better residual value %f!\n',its,res);
        model.gamma = gamma;
        model.beta = beta;
        model.tau = tau;
        model.G = G;
        res_opt = res;
    end    
    
end 

model.params = params;
end

function [G,tau,beta] = forwardpass(gamma,params)
%Compute the model elements
%Inputs
%   gamma        :  m hyper parameters for m-1 beta - parameters of the mixture of sigmoids
%   params.m     :  number of segments
%   params.alpha :  shared parameter of the mixture of sigmoids
%   params.lambda:  paramter control balance regularization term
%Outputs
%   beta         :  paramters of the mixture of sigmoids (1 x (m-1)) (change points)
%   tau          :  sample-to-segment indicator (1 x n)
%   G            :  sample-to-segment indicator matrix (m x n)

%% compute beta
e_gamma = exp(gamma);
beta =  1 + (params.n-1) * cumsum(e_gamma) / sum(e_gamma);
beta(end) = []; % remove the last element because beta(end) = n

%% compute tau
tau = sigmoid_mixture(1:1:params.n,params.alpha,beta);

%% approximate G
G = max(0,1 - abs(tau-(1:1:params.m)')); 

end

function grad = backwardpass(idxs,K,G,tau,beta,gamma,params)
%Compute gradient of the objective w.r.t gamma
%Inputs
%   idxs: indexes of the samples in the minibatch 1 x b
%   K: minibatch kernel matrix b x b
%   G: current minibatch indicator matrix m x b
%   tau: current minibatch sample-segment indicator 1 x b
%   beta: current m-1 paramters of the mixture of sigmoids
%   gamma: current m hyper parameters for m-1 beta
%   params
%       params.alpha    : shared parameter of the mixture of sigmoids
%       params.lambda   : paramter controls balance pernalty
%       params.n        : total number of samples
%       params.m        : number of segments
%       params.batchsize: minibatch size

%% compute dJ_dG
GGT_invG = ((G * G')^-1)*G;
dJ_dG = 2 * (GGT_invG*K*(eye(params.batchsize) - G'*GGT_invG) - params.lambda*G*ones(params.batchsize));

%% compute dJ_dtau
dG_dtau = tau - (1:1:params.m)';
dG_dtau = (dG_dtau > -1 & dG_dtau < 0) - (dG_dtau >= 0 & dG_dtau < 1);
dJ_dtau = sum(dJ_dG .* dG_dtau,1); % 1 x b

%% compute dJ_dbeta
dtau_dbeta = zeros(params.batchsize,params.m-1);
for j=1:params.m-1
    dtau_dbeta(:,j) = sigmoid(idxs,params.alpha,beta(j))';
end
dtau_dbeta = params.alpha * (dtau_dbeta .* (dtau_dbeta - 1));
dJ_dbeta = dJ_dtau * dtau_dbeta; % 1 x (m - 1)

%% compute dJ_dgamma
e_gamma = exp(gamma);
ega_sum = sum(e_gamma);
add = tril(((params.n-1) / ega_sum) * ones(1,params.m)' * e_gamma);
dbeta_dgamma = (1-params.n) * (cumsum(e_gamma) / (ega_sum^2))' * e_gamma + add;
dbeta_dgamma(end,:) = []; % delete the last row 
grad = dJ_dbeta * dbeta_dgamma;

end

function eta = backtracking(idxs,K,gamma,grad,params)
%Estimate stepsize for the stochastic gradient 
%Inputs
%   idxs  : indexes of samples in the minibatch
%   K     : kernel matrix corresponding with the samples in the minibatch
%   gamma : current gamma
%   grad  : stochastic gradient
%Outputs
%   eta   : step size for the stochastic gradient

eta = params.eta_init;
decay = 0.5;

obj_new = intmin('int64');
obj = forward_minibatch(idxs,K,gamma,params);

its = 0;
while (obj_new <= (obj - eta * 1e-6 * grad * grad') && its < 20) || isnan(obj_new)
    eta = decay * eta;
    gamma_new = gamma + eta * grad;
    [obj_new,~,~,~] = forward_minibatch(idxs,K,gamma_new,params);
    its = its + 1;
end

end

function [obj,G,tau,beta] = forward_minibatch(idxs,K,gamma,params)
%Compute objective funtion value w.r.t samples in the minibatch
%Inputs
%   idxs  : indexes of samples in the minibatch
%   K     : kernel matrix corresponding with the samples in the minibatch
%   gamma : current gamma
%Outputs
%   beta  : corrspoding beta for current gamma
%   tau   : corrspoding tau of samples in the minibatch for current gamma
%   G     : corrspoding G of samples in the minibatch for current gamma

%% compute beta
e_gamma = exp(gamma);
beta =  1 + (params.n-1) * cumsum(e_gamma) / sum(e_gamma);
beta(end) = []; % remove the last element because beta(end) = n

%% compute tau
tau = sigmoid_mixture(idxs,params.alpha,beta);

%% approximate G
G = max(0,1 - abs(tau-(1:1:params.m)')); 

%% compute the objective
obj = trace(G'*((G*G')^-1)*G*K) - params.lambda*trace(G'*G*ones(params.batchsize));

end

