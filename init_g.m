function gamma = init_g(m,seed)
%generate gamma randomly
rng(seed);
mu = ones(1,m);
sigma = 0.25;
gamma = normrnd(mu,sigma);
end