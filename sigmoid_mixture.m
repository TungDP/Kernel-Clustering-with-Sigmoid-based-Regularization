function val = sigmoid_mixture(x,c_1,c_2)
%mixture of sigmoids
%   x: input
%   c_1: steepness of the curve (shared among sigmoids)
%   c_2: vectors of midpoints (increasing constraint, c_2(1) = 0.5)
m = length(c_2); % number of sigmoids
val = ones(size(x));
for i=1:m
    val = val + sigmoid(x,c_1,c_2(i));
end
end
