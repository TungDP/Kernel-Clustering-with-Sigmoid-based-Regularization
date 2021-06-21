function val = sigmoid_mixture_cutoff(x,c_1,c_2,cutoff)
%mixture of sigmoids with cutoff
%   x      : input (concatenation of s sequences)
%   c_1    : steepness of the curve (shared among sigmoids)
%   c_2    : vectors of midpoints [1 x sk-s]
%   cutoff : cutoff points [1 x s]
%% parameters
m = length(c_2);                % number of sigmoids
s = numel(cutoff) + 1;          % number of sequences (number cutoff points + 1)
k = round((m + s) / s);         % number of segments
%% compute value
val = sigmoid_mixture(x,c_1,c_2) + (1-k) * (sigmoid_mixture(x,c_1,cutoff) - 1);    
end