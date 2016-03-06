function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim
% 두 벡터의 유사한 정도를 수치화한다.

% Ensure that x1 and x2 are column vectors
% 컬럼벡터로 만든다. 
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
sim = exp( -1 * sum((x1 - x2).^2) / (2*sigma^2) );





% =============================================================
    
end
