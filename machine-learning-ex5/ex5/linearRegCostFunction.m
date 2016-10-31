function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% theta is a 2 x 1 vector
prediction = X * theta ;  % 12 x 1 vector


J =  1/(2 * m) * sum( (prediction - y) .^ 2) + lambda/(2 * m) * sum(theta([2:size(theta,1)]) .^ 2);

%grad = [   ; ]; 
grad(1) = 1/m * ((prediction - y)' * X(:,1));
grad([2:size(theta,1)]) = (1/m * ((prediction - y)' * X (:,[2:size(X,2)])) + (lambda/m) * theta([2:size(theta,1)])')'  ; 

% the above implementation is overly complicated, basically this should yield a vector of size equaling total number of inputs. in this case there was just one... 
% ... input so the calculations were correct - but the dimensions were not. in case there were more the calculations would be wrong. the right way to do this is to actually 
% ...vectorize this in a way such that you get predictionerror(1)*x(11)+ predictionerror(2) * x(12) +...predictionerror(m)*x(1m)
%....here prediction(k) is the error for the kth sample, and  x(ik) is the kth sample of ith input. k ofcourse ranges from 1 to m

%grad([2:size(X,2)],1) = m^-1. * (X(:,[2:size(X,2)])' * (prediction-y)) + (lambda/m)*theta([2:size(X,2)]); this is the right way to calculate gradients

% =========================================================================

grad = grad(:) ; 

end
