function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% size(Theta1): 25 x [dim(X) + 1]
% i.e. there are dim(X)+1 inputs (including x0 = 1), leading to
%      25 neurons in the first hidden layer

z2 = Theta1(:,2:end) * X' + Theta1(:,1);
a2 = sigmoid(z2);

% size(Theta2): num_class x 26
% i.e. 26 inputs (the 25 previous, plus the bias term)
% from a2 to num_class outputs
z3 = Theta2(:,2:end) * a2 + Theta2(:,1);
a3 = sigmoid(z3);

[probability, p] = max(a3);







% =========================================================================


end
