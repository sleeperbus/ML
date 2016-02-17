function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network



Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%fprintf('\nSize of X: (%d x %d)\n', size(X, 1), size(X, 2));
%fprintf('\nSize of y: (%d x %d)\n', size(y, 1), size(y, 2));
%fprintf('\nSize of Theta1: (%d x %d)\n', size(Theta1, 1), size(Theta1, 2));
%fprintf('\nSize of Theta2: (%d x %d)\n', size(Theta2, 1), size(Theta2, 2));



% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% X 에 dummy feature를 더한다.
X = [ones(m, 1) X];

% y 는 1부터 10까지 들어온다. 10인 경우는 0을 나타낸다. y 를 vector 로 변환되어야 한다. 
% y = m * num_labels
new_y = zeros(m, num_labels);
for i = 1:num_labels
	num_list = find(y == i);
	new_y(num_list, i) = 1;
end;
y = new_y;

% 입력된 Theta 를 사용해 이론값을 구한다. 
% result = m * num_labels
result =  sigmoid([ones(m, 1) (sigmoid(Theta1 * X'))'] * Theta2');
J = -1 * (sum(sum(y .* log(result) + (1 - y) .* log(1 - result))) / m);

% regularization 을 수행한다. 
t_theta1 = Theta1(:, 2:end);
t_theta2 = Theta2(:, 2:end);
J = J + ...
	( ...
		sum((t_theta1 * t_theta1' .* eye(size(t_theta1, 1)))(:)) + ...
		sum((t_theta2 * t_theta2' .* eye(size(t_theta2, 1)))(:))
	) * lambda /(2*m);


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% 우선은 loop로 backprop 을 구현
D_1 = zeros(size(Theta1));
D_2 = zeros(size(Theta2));
% for t = 1:m 
% 	% forward pass
% 	% (10 x 1)
% 	y_vec = y(t, :)';

% 	% (401 x 1)
% 	a_1 = X(t, :)';

% 	% (25 x 401) * (401 x 1) = (25 x 1)
% 	z_2 = Theta1 * a_1;

% 	% (26 x 1)
% 	a_2 = [1; sigmoid(z_2)];

% 	% (10 x 26) * (26 x 1) = (10 x 1)
% 	z_3 = Theta2 * a_2;

% 	% (10 x 1)	
% 	a_3 = sigmoid(z_3);

% 	% 여기서부터 backprop
% 	% (10 x 1)
% 	d_3 = a_3 - y_vec;

% 	% (10 x 26)' * (10 x 1) .* (26 x 1) = (25 x 1)
% 	d_2 = (Theta2' * d_3)(2:end) .* sigmoidGradient(z_2);

% 	% (25 x 1) * (401 x 1)' = (25 x 401)
% 	D_1 = D_1 + d_2 * a_1';

% 	% (10 x 1) * (26 x 1)' = (10 x 26)
% 	D_2 = D_2 + d_3 * a_2';
% end	
% Theta1_grad = 1/m * D_1;
% Theta2_grad = 1/m * D_2;


% 여기서부터는 matrix 연산으로 구현한 NN

% 처음은 역시 feed forward부터 시작한다. 
% (5000 x 401)
a_1 = X;
%fprintf('\nSize of a_1: (%d x %d)\n', size(a_1, 1), size(a_1, 2));

% (5000 x 401) *  (401 x 25) = (5000 x 25)
z_2 = a_1 * Theta1';
%fprintf('\nSize of z_2: (%d x %d)\n', size(z_2, 1), size(z_2, 2));

% (5000 x 26) 
a_2 = [ones(m,1) sigmoid(z_2)]; 
%fprintf('\nSize of a_2: (%d x %d)\n', size(a_2, 1), size(a_2, 2));

% (5000 x 26) * (26 * 10) = (5000 x 10)
z_3 = a_2 * Theta2';
%fprintf('\nSize of z_3: (%d x %d)\n', size(z_3, 1), size(z_3, 2));

% (5000 x 10)
a_3 = sigmoid(z_3);
%fprintf('\nSize of a_3: (%d x %d)\n', size(a_3, 1), size(a_3, 2));

% 여기서부터는 backprop
% (5000 x 10)
d_3 = a_3 .- y;
%fprintf('\nSize of d_3: (%d x %d)\n', size(d_3, 1), size(d_3, 2));

% ((5000 x 10) * (10 x 26))(:, 2:end) .* (5000 x 25) = (5000 x 25)
d_2 = ((d_3 * Theta2))(:, 2:end) .* sigmoidGradient(z_2);
%fprintf('\nSize of d_3: (%d x %d)\n', size(d_3, 1), size(d_3, 2));

% (5000 x 25)' * (5000 x 401) = (25 x 401)
D_1 = d_2' * a_1;

% (5000 x 10)' * (5000 x 26) = (10 x 26)
D_2 = d_3' * a_2;

Theta1_grad = 1/m * D_1;
Theta2_grad = 1/m * D_2;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
