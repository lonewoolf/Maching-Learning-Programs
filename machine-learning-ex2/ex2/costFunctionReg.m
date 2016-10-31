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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% theta([2:size(X,2)]  means that we dont want to regularize theta0

prediction = sigmoid(X*theta);
J = 1/m * sum((-1.* y)' * log(prediction) - (1. -y)' * log(1. -prediction)) + lambda/(2*m) * sum(theta([2:size(X,2)]).^2);

grad(1,1) = (X(:,1)' * (prediction-y))/m;
%grad 2-28
grad([2:size(X,2)],1) = m^-1. *(X(:,[2:size(X,2)])' * (prediction-y)) + (lambda/m)*theta([2:size(X,2)]);



% =============================================================

end
