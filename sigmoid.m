function val = sigmoid(x,c_1,c_2)
% sigmod function
%   x  : input
%   c_1: steepness of the curve
%   c_2: midpoint 
val = 1 ./ (1 + exp(c_1 * (c_2 - x)));
end