function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));  % theta is a 28 dimensional vector

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

prediction = sigmoid(X*theta);

J = 1/m * sum((-1.* y)' * log(prediction) - (1. -y)' * log(1. -prediction))

%gradient descent 
%(prediction-y) [100X1] dimensional vector
% X  [100 X 3] matrix
%grad will be  a [3 x1] vector ( for 3 parameters)

grad=  m^-1. *(X' * (prediction-y))
% =============================================================

end
