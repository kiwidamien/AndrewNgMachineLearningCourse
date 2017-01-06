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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% Part 1: Feed forward

z2 = X*Theta1(:,2:end)' + Theta1(:,1)';  % is m x 25
a2 = sigmoid(z2);

z3 = a2*Theta2(:,2:end)' + Theta2(:,1)'; % is m x 10 
h = sigmoid(z3);                         % is m x 10, the outputs


J = 0;

% Can we vectorize over examples, and loop over classes
% instead of vectorizing over classes and looping over examples?

%lh = log(h + 1e-99);
%lhc= log(1 - h + 1e-99);

%for k = 1:num_labels,
%  yk = (y == k);
%  J = J + yk'*lh + (1-yk)'*lhc;
%end
%J = -sum(J)/m;

for example = 1:m,
  my_y = zeros(1,num_labels);
  my_y(y(example)) = 1;
  J = J + my_y*log(h(example,:) + 1e-99)' + (1-my_y)*log(1-h(example,:) + 1e-99)';
end
J = -J/m;

% Flatten the Theta parameters, excluding the row corresponding to the bias
% term.

temp = [Theta1(:,2:end)(:); Theta2(:,2:end)(:)];
J = J + lambda/(2*m)*temp'*temp;

% Part 2: Implementing the backpropagation algorithm

for example = 1:m,
  yk = 1:num_labels;
  yk = (yk == y(example));  % Set yk to be 1 at location y(example), 0 otherwise
  delta3 = h(example,:) - yk;

  %delta2 = Theta2'* delta3' .* sigmoidGradient(z2(example,:));
  delta2 = (delta3 * Theta2) .* sigmoidGradient([sigmoid(1) z2(example,:)]);
  
  delta2 = delta2(2:end);   % Removes "error" associated with bias term

  Theta2_grad = Theta2_grad + delta3'*[1  a2(example,:)];
  Theta1_grad = Theta1_grad + delta2'*[1   X(example,:)];
end

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;  

% Regularize, excluding the bias parameter
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
