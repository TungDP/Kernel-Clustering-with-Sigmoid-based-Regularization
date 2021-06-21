function model = KCSR_balanced_Multi_SGAm(Xs,params)
%Inputs
%   Xs      :  data sequences {X1 X2 ... Xk} (d x n)
%   params  :
%       params.m        :  number of segments per sequence
%       params.alpha    :  shared paramter of the mixture of sigmoids
%       params.s        :  sigma for kernel function
%       params.kn       :  kernel function
%       params.lambda   :  paramter control balance regularization term
%       params.batchsize:  minibatch size
%       params.maxepoch :  number of time pass over the sequence
%       params.eta_0    :  initial step size
%       params.decay    :  the geometric rate in which the learning rate decays.
%       params.momentum :  momentum parameter for SGA, with value in [0 1).
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
%       model.objs  :  optimal objective at each iteration [trace(G'*((G *
%       G')^-1)*G*K) - lambda*trace(G'*G*1*1')]
%       model.times :  time spent for each epoch
%
%       model.params:  model paramters
%           model.params.m        :  number of segments
%           model.params.alpha    :  shared paramter of the mixture of sigmoids
%           model.params.s        :  sigma for kernel function
%           model.params.kn       :  kernel function
%           model.params.lambda   :  paramter control balance regularization term
%           model.params.cutoff   :  cuttoff points
%           model.params.batchsize:  minibatch size
%           model.params.maxepoch :  number of time pass over the sequence
%           model.params.eta_0    :  initial step size
%           model.params.decay    :  the geometric rate in which the learning rate decays.
%           model.params.momentum :  momentum parameter for SGD, with value in [0 1).

%% preprocess input sequences
k = length(Xs); % number of sequences
cutoff = zeros(1,k); X = [];
for i=1:k
    % collect cutoff points
    cutoff(i) = size(Xs{i},2);
    % concatenate sequences
    X = [X Xs{i}];
end
cutoff = cumsum(cutoff);
params.n = cutoff(k);
params.cutoff = cutoff(1:k-1) + 0.5;
params.k = k;

%% compute kernel matrix
fprintf('compute kernel matrix ...\n');
K = params.kn(X,X,params.s);

%% initialize gamma
gamma = init_g(k * params.m,params.seed);
fprintf('initializing gamma ...\n');
for i = 1:k
    fprintf('sequence %d: initial gamma = %s\n', ...
            i,num2str(gamma(1+(i-1)*params.m:i*params.m)));
end

%% save initial model
[obj,G,tau,beta] = forwardpass(K,gamma,params);

model.gamma_init = gamma;
model.beta_init = beta;
model.tau_init = tau;
model.G_init = G;

model.gamma = gamma;
model.beta = beta;
model.tau = tau;
model.G = G;
model.objs = [obj];
model.times = [];

%% run stochastic gradient ascent
fprintf('initial beta = %s\n',num2str(beta));
fprintf('initial objective value = %.5f\n',model.objs(end));
fprintf('start stochastic gradient ascent ...\n');

numbatches = ceil(params.n/params.batchsize);
its=0; delta=0; obj_opt = obj;
while its < params.maxepoch
    
    eta_init = params.eta_0 * params.decay ^ its; % Reduce learning rate.
    params.eta_init = eta_init;      
    t0=tic;
    rp = randperm(params.n); % Shuffle the data set.
    for i=1:numbatches
        % extract minibatch
        start = (i-1)*params.batchsize+1;
        stop  = min(i*params.batchsize,params.n);
        idxs  = sort([rp(start:stop),rp(1:max(0,i*params.batchsize-params.n))]);
        % estimate stochastic gradient
        grad = backwardpass(idxs,K(idxs,idxs),G(:,idxs),tau(idxs),beta,gamma,params);
        % estimate step size
        eta = backtracking(idxs,K(idxs,idxs),gamma,grad,params);
        % update gamma
        delta = eta * grad; % - params.momentum * delta;
        gamma = gamma + delta;
    end
    
    %% record the time spent for each epoch.
    its=its+1; model.times=[model.times, toc(t0)]; params = rmfield(params,'eta_init');
    
    %% check objective
    [obj,G,tau,beta] = forwardpass(K,gamma,params);
    model.objs = [model.objs obj]; % save objective value
    if obj > obj_opt
        % save model if objective is improved
        fprintf('epoch %d: getting better objective value %f!\n',its,obj);
        model.gamma = gamma;
        model.beta = beta;
        model.tau = tau;
        model.G = G;
        obj_opt = obj;
    end
    
end

model.params = params;

end

function [obj,G,tau,beta] = forwardpass(K,gamma,params)
%Compute the model elements
%Inputs
%   K            :  kernel matrix (n x n)
%   gamma        :  km hyper parameters for k(m-1) beta - parameters of the
%   mixture of sigmoids cutoff
%   params
%       params.n     :  number of total samples
%       params.m     :  number of segments per subsequence
%       params.k     :  number of subsequences
%       params.alpha :  shared parameter of the mixture of sigmoids
%       params.cutoff:  cutoff points for the mixture of sigmoids
%       params.lambda:  paramter control balance regularization term
%Outputs
%   beta         :  paramters of the mixture of sigmoids (1 x k(m-1)) (change points)
%   tau          :  sample-to-segment indicator (1 x n)
%   G            :  sample-to-segment indicator matrix (m x n)
%   obj          :  objective value

%% parameter
stops = [0, params.cutoff - 0.5, params.n]; % stop (start and end) points  of subsequences

%% compute beta
e_gamma = exp(gamma); beta = [];
for i=1:params.k
    e_gamma_sub = e_gamma(1+(i-1)*params.m:i*params.m);
    beta_sub = stops(i) + 1 + (stops(i+1) - stops(i) - 1) * cumsum(e_gamma_sub) / sum(e_gamma_sub);
    beta_sub(end) = []; beta = [beta beta_sub];
end  

%% compute tau
tau = sigmoid_mixture_cutoff(1:1:params.n,params.alpha,beta,params.cutoff);

%% approximate G
G = max(0,1 - abs(tau-(1:1:params.m)')); 

%% compute the objective
obj = trace(G'*((G*G')^-1)*G*K) - params.lambda*trace(G'*G*ones(params.n));

end

function grad = backwardpass(idxs,K,G,tau,beta,gamma,params)
%Compute gradient of the objective w.r.t gamma
%Inputs
%   idxs: indexes of the samples in the minibatch 1 x b
%   K: minibatch kernel matrix b x b
%   G: current minibatch indicator matrix m x b
%   tau: current minibatch sample-segment indicator 1 x b
%   beta: current k(m-1) paramters of the mixture of sigmoids cutoff
%   gamma: current km hyper parameters for k(m-1) beta
%   params
%       params.alpha    : shared parameter of the mixture of sigmoids
%       params.lambda   : paramter controls balance pernalty
%       params.batchsize: size of the minibatch
%Outputs
%   grad: stochastic gradient w.r.t gamma (1 x km)

%% parameters
stops = [0, params.cutoff - 0.5, params.n]; % stop (start and end) points of subsequences

%% compute dJ_dG
GGT_invG = ((G * G')^-1)*G; % O(m^2 x b)
dJ_dG = 2 * (GGT_invG*K*(eye(params.batchsize) - G'*GGT_invG) - params.lambda*G*ones(params.batchsize)); % O(m x b^2)

%% compute dJ_dtau
dG_dtau = tau - (1:1:params.m)'; % O(m x b)
dG_dtau = (dG_dtau > -1 & dG_dtau < 0) - (dG_dtau >= 0 & dG_dtau < 1); % O(m x b)
dJ_dtau = sum(dJ_dG .* dG_dtau,1); % 1 x b % O(m x b)

%% compute dJ_dbeta
dtau_dbeta = zeros(params.batchsize,params.k*(params.m-1)); % batchsize x k(m-1)
for j=1:params.k*(params.m-1)
    dtau_dbeta(:,j) = sigmoid(idxs,params.alpha,beta(j))';
end
dtau_dbeta = params.alpha * (dtau_dbeta .* (dtau_dbeta - 1));
dJ_dbeta = dJ_dtau * dtau_dbeta; % 1 x k(m - 1) % O(m x b)

%% compute dJ_dgamma
e_gamma = exp(gamma); dbeta_dgamma = zeros(params.k*(params.m - 1),params.k*params.m); % k(m-1) x km
for i=1:params.k
    e_gamma_sub = e_gamma(1+(i-1)*params.m:i*params.m);
    ega_sum_sub = sum(e_gamma_sub);
    add_sub = tril(((stops(i+1) - stops(i) - 1) / ega_sum_sub) * ones(1,params.m)' * e_gamma_sub);
    dbeta_dgamma_sub = (stops(i) + 1 - stops(i+1)) * (cumsum(e_gamma_sub) / (ega_sum_sub^2))' * e_gamma_sub + add_sub;
    dbeta_dgamma_sub(end,:) = []; % delete the last row
    dbeta_dgamma(1+(i-1)*(params.m-1):i*(params.m-1),1+(i-1)*params.m:i*params.m) = dbeta_dgamma_sub;
end
grad = dJ_dbeta * dbeta_dgamma; % 1 x km % O(m^2)

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
%   params
%       params.n        : total number of samples
%       params.m        : number of segments per subsequence
%       params.k        : number of subsequences
%       params.alpha    : shared parameter of the mixture of sigmoids
%       params.cutoff   : cutoff points for the mixture of sigmoids
%       params.lambda   : paramter controls balance pernalty
%       params.batchsize: size of the minibatch
%Outputs
%   beta  : corrspoding beta for current gamma
%   tau   : corrspoding tau of samples in the minibatch for current gamma
%   G     : corrspoding G of samples in the minibatch for current gamma

%% parameter
stops = [0, params.cutoff - 0.5, params.n]; % stop (start and end) points  of subsequences

%% compute beta
e_gamma = exp(gamma); beta = [];
for i=1:params.k
    e_gamma_sub = e_gamma(1+(i-1)*params.m:i*params.m);
    beta_sub = stops(i) + 1 + (stops(i+1) - stops(i) - 1) * cumsum(e_gamma_sub) / sum(e_gamma_sub);
    beta_sub(end) = []; beta = [beta beta_sub];
end 

%% compute tau
tau = sigmoid_mixture_cutoff(idxs,params.alpha,beta,params.cutoff);

%% approximate G
G = max(0,1 - abs(tau-(1:1:params.m)')); 

%% compute the objective
obj = trace(G'*((G*G')^-1)*G*K) - params.lambda*trace(G'*G*ones(params.batchsize));

end
