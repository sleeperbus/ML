function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
% training 에 사용되는 데이터의 개수를 1부터 시작해서 size(X, 1) 까지 늘린다. 
% 즉 size(X, 1) 만큼의 theta 들이 생성되게 된다. 
% 이것들을 이용해서 train error cost 와 cross-validation error cost 를 구한다. 
% train error 를 구할 때는 해당 theta 가 구해진 size 에 대해서만 구하고 
% cv error 를 구할 때는 전체 cv set 에 대해서 구해야 한다. 
% cost 를 구할 때는 function 에 넘기는 lambda 를 0으로 설정해야 한다. 

for i = 1:m 
	theta = trainLinearReg(X(1:i, :), y(1:i), lambda);	
	trainError = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
	cvError = linearRegCostFunction(Xval, yval, theta, 0);
	error_train(i) = trainError;
	error_val(i) = cvError;
end

% -------------------------------------------------------------

% =========================================================================

end
