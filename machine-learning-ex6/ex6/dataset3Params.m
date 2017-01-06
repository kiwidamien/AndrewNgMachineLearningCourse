function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


samples = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
k       = @(x1,x2) (gaussianKernel(x1,x2,sigma));
model   = svmTrain(X, y, C, k, 1e-3, 20);
smallError = mean(double( svmPredict(model, Xval) ~= yval));

for Ctest = samples
  for sigmatest = samples
    printf("C = %4.2f,  sigma =%4.2f  (best values %4.2f and %4.2f with error %6.4f)", Ctest, sigmatest,C,sigma, smallError)
    kernel= @(x1,x2)(gaussianKernel(x1,x2,sigmatest));
    model = svmTrain(X,y,Ctest, kernel, 1e-3,10);
    error = mean(double( svmPredict(model, Xval) ~= yval ) );
    if error < smallError;
      smallError = error;
      C = Ctest;
      sigma = sigmatest;
    end
  end
end



% =========================================================================

end
