function model = KCSR_balanced_FB(X,params)
%Inputs
%   X       :  data sequence (d x n)
%   params  :
%       params.m        :  number of segments
%       params.alpha    :  shared paramter of the mixture of sigmoids
%       params.s        :  sigma for kernel function
%       params.kn       :  kernel function
%       params.lambda   :  paramter control balance regularization term
%       params.eta_0    :  initial step size
%       params.tolerance:  convergence criterion 
%Outputs
%   model   :
%       model.gamma_init
%       model.beta_init
%       model.tau_init
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
%           model.params.m     :  number of segments
%           model.params.alpha :  shared paramter of the mixture of sigmoids
%           model.params.s     :  sigma for kernel function
%           model.params.kn    :  kernel function
%           model.params.lambda:  paramter control balance regularization term

%% compute kernel matrix
fprintf('compute kernel matrix ...\n');
K = params.kn(X,X,params.s);

%% initialize gamma
gamma = init_g(params.m,params.seed);
fprintf('initial gamma = %s\n',num2str(gamma));

%% full-batch gradient ascent
[obj,G,tau,beta] = forwardpass(K,gamma,params);
obj_old = obj;

% save initial model
model.gamma_init = gamma;
model.beta_init = beta;
model.tau_init = tau;
model.G_init = G;
model.objs = [obj];
model.times = [];

% run
fprintf('start full-batch gradient ascent ...\n');
fprintf('initial objective value = %.5f\n',obj_old);
for i=1:1e4
    t0=tic; % start time counting 
    grad = backwardpass(K,G,tau,beta,gamma,params);
    [obj,G,tau,beta,gamma] = backtracking(obj,K,gamma,grad,params);
    fprintf('iteration %d, obj = %.5f\n',i,obj);
    model.times=[model.times, toc(t0)]; model.objs = [model.objs obj]; 
    if abs(obj - obj_old) <= params.tolerance
        break;
    end
    obj_old = obj;
end

% save model
fprintf('save the model ...\n');
model.gamma = gamma;
model.beta = beta;
model.tau = tau;
model.G = G;
model.params = params;

end

function [obj_new,G_new,tau_new,beta_new,gamma_new] = backtracking(obj,K,gamma,grad,params)

eta = params.eta_0;
tau = 0.5;

obj_new = intmin('int64');
while obj_new <= (obj - eta * 1e-6 * grad * grad') || isnan(obj_new)
    eta = tau * eta;
    gamma_new = gamma + eta * grad;
    [obj_new,G_new,tau_new,beta_new] = forwardpass(K,gamma_new,params);
end

end

function [obj,G,tau,beta] = forwardpass(K,gamma,params)
%Compute the objective and approximation of the indicator matrix
%   K: kernel matrix n x n
%   gamma:          m hyper parameters for m-1 beta - parameters of the mixture of sigmoids
%   params.alpha:   shared parameter of the mixture of sigmoids
%   params.lambda:  paramter control balance regularization term

%% parameter
[n,~] = size(K); % number of samples 
m = length(gamma); % number of segments (gammas)

%% compute beta
e_gamma = exp(gamma);
beta =  1 + (n-1) * cumsum(e_gamma) / sum(e_gamma);
beta(end) = []; % remove the last element because beta(end) = n

%% compute tau
tau = sigmoid_mixture(1:1:n,params.alpha,beta);

%% approximate G
G = max(0,1 - abs(tau-(1:1:m)'));   

%% compute the objective
obj = trace(G'*((G*G')^-1)*G*K) - params.lambda*trace(G'*G*ones(n));

end

function grad = backwardpass(K,G,tau,beta,gamma,params)
%Compute gradeint of the objective w.r.t gamma
%   K: kernel matrix n x n
%   G: current indicator matrix m x n
%   tau: current sample-segment indicator 1 x n
%   beta: current m-1 paramters of the mixture of sigmoids
%   gamma: current m hyper parameters for m-1 beta
%   params.alpha: shared parameter of the mixture of sigmoids
%   params.lambda: paramter controls balance pernalty

%% parameters
n = numel(tau); % sequence length
m = numel(gamma); % number of segments

%% compute dJ_dG
GGT_invG = ((G * G')^-1)*G;
dJ_dG = 2 * (GGT_invG*K*(eye(n) - G'*GGT_invG) - params.lambda*G*ones(n));

%% compute dJ_dtau
dG_dtau = tau - (1:1:m)';
dG_dtau = (dG_dtau > -1 & dG_dtau < 0) - (dG_dtau >= 0 & dG_dtau < 1);
dJ_dtau = sum(dJ_dG .* dG_dtau,1); % 1 x n

%% compute dJ_dbeta
dtau_dbeta = zeros(n,m-1);
for j=1:m-1
    dtau_dbeta(:,j) = sigmoid(1:1:n,params.alpha,beta(j))';
end
dtau_dbeta = params.alpha * (dtau_dbeta .* (dtau_dbeta - 1));
dJ_dbeta = dJ_dtau * dtau_dbeta; % 1 x (m - 1)

%% compute dJ_dgamma
e_gamma = exp(gamma);
ega_sum = sum(e_gamma);
add = tril(((n-1) / ega_sum) * ones(1,m)' * e_gamma);
dbeta_dgamma = (1-n) * (cumsum(e_gamma) / (ega_sum^2))' * e_gamma + add;
dbeta_dgamma(end,:) = []; % delete the last row 
grad = dJ_dbeta * dbeta_dgamma;

end