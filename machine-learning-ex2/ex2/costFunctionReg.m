function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
mask = ones(size(theta));
mask(1) = 0;

J = (-1/m) * (y' * log(sigmoid(X*theta)) + (1-y)' * log(1 - sigmoid(X*theta))) + (theta'*theta - theta(1)^2) * lambda / 2 / m; 
grad = (1/m) * ((sigmoid(X * theta) - y)' * X)' + mask .* theta * lambda / m;

% =============================================================

end
