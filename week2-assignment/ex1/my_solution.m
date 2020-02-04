data = load('ex1data1.txt');
x = data(:, 1);
y = data(:, 2);
m = length(y);
% plot(x, y, 'rx', 'MarkerSize', 10);
x = [ones(m,1), data(:,1)];
theta = zeros(2, 1);
iterations = 1500;
alpha = 0.01;

function J = computeCost(x, y, theta)
  %COMPUTECOST Compute cost for linear regression
  %   J = COMPUTECOST(x, y, theta) computes the cost of using theta as the
  %   parameter for linear regression to fit the data points in X and y

  % Initialize some useful values
  m = length(y); % number of training examples
  % You need to return the following variables correctly 
  J = 0;

  % Instructions: Compute the cost of a particular choice of theta
  %               You should set J to the cost.
  h = x * theta;
  J = (1/(2*m) * sum((h - y) .^ 2));
end

J = computeCost(x, y, theta);
disp(J);

function [theta, J_history] = gradientDescent(x, y, theta, alpha, iterations)
  m = length(y); % number of training examples
  J_history = zeros(iterations, 1);

  for iter = 1:iterations
    error = x * theta - y;
    temp1 = theta(1) - (alpha/m) * sum (error);
    temp2 = theta(2) - (alpha/m) * sum (error .* x(:,2));
    theta = [temp1; temp2];

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(x, y, theta);
  end

end

theta = gradientDescent(x,y,theta,alpha, iterations);
disp(theta);