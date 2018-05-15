% Distributed SCAD regularized logistic regression
% (compared against SCAD_logreg package)
clc;
clear all;
%% Generate problem data
% rand('seed', 0);
% randn('seed', 0);

n1 = 150;  % the number of samples;
n2 = 150;
n3 = 200;
n = n1 + n2 + n3; 
p = 20;  % the number of features
N = 3;  % the number of agent;

beta_int=zeros(1,p)';
beta_int(1)=1.5;
beta_int(2)=-1.2;
beta_int(3)=1.8;
beta_int(4)=-2;
beta_int(5)=2.5;
beta_int(6)=-1.2;
beta_int(7)=1;
beta_int(8)=-1.5;
beta_int(9)=2;
beta_int(10)=-1.6;
beta_zero = randn(1);                   % random intercept
beta_true = [beta_zero; beta_int];

%% simulate dataset1
X1 = sprandn(n1,p,0.1);
% Setting correlation
cor=0.2;             % correlation ¦Ñ=0.2, 0.4
for i=1:n1
    for j=2:p
        X1(i,j) =  X1(1,1) * cor + X1(i,j) * (1-cor);
    end
end
sigm1 = 0.1;         % noise sigm= 0.1, 0.3, 0.5
Y1 = sign( X1 * beta_int + beta_zero + sigm1 * normrnd(0, 1, n1, 1));

%% simulate dataset2
X2 = sprandn(n2,p,0.1);
% Setting correlation
cor=0.2;             % correlation ¦Ñ=0.2, 0.4
for i=1:n2
    for j=2:p
        X2(i,j) =  X2(1,1) * cor + X2(i,j) * (1-cor);
    end
end
sigm2 = 0.2;         % noise sigm=0.2, 0.4, 0.6
Y2 = sign( X2 * beta_int + beta_zero + sigm2 * normrnd(0, 1, n2, 1));

%% simulate dataset3
X3 = sprandn(n3,p,0.1);
% Setting correlation
cor=0.2;             % correlation ¦Ñ=0.2, 0.4
for i=1:n3
    for j=2:p
        X3(i,j) =  X3(1,1) * cor + X3(i,j) * (1-cor);
    end
end
sigm3 = 0.3;         % noise sigm=0.3, 0.5, 0.7
Y3 = sign( X3 * beta_int + beta_zero + sigm3 * normrnd(0, 1, n3, 1));

%%
X0=[X1;X2;X3];
Y0=[Y1;Y2;Y3];
Y0_origin=Y0;

% packs all observations in to an m*N x n matrix
A0 = spdiags(Y0, 0, n, n) * X0;
ratio = sum(Y0 == 1)/(n);

%% Setting lambda && Cross validation
lambda_max = 1/(n) * norm((1-ratio)*sum(A0(Y0==1,:),1) + ratio*sum(A0(Y0==-1,:),1), 'inf');
lambda_min = lambda_max * 0.1;
m = 10;
for i=1:m
    lambda(i) = lambda_max*(lambda_min/lambda_max)^(i/m);
    [beta history] = distr_SCAD_logreg(A0, Y0, lambda(i), N, 1.0, 1.0, n1, n2, n3);   % (X, Y, lambda, N, rho, alpha)
    beta_path(:,i)=beta; 
    i
end

[opt,Mse]=CV_distri_SCAD_logistic(A0, Y0, lambda, N, beta_int, beta_zero, n1, n2, n3);
beta=beta_path(:,opt);

%% without Cross validation
% lambda = 0.1 * 1/(n) * norm((1-ratio)*sum(A0(Y0==1,:),1) + ratio*sum(A0(Y0==-1,:),1), 'inf');
% [beta history] = distr_SCAD_logreg(A0, Y0, lambda, N, 1.0, 1.0, n1, n2, n3);   % (X, Y, lambda, N, rho, alpha)

%% Solve problem
% generate testing data %
n_test=200;
X_test = randn(n_test, p); 
sigm=0.2;         % noise = 0.2
l_test = X_test * beta_int + beta_zero + sigm * normrnd(0, 1, n_test, 1);
prob_test=exp(l_test)./(1 + exp(l_test));
for i=1:n_test
    if prob_test(i)>0.5
        Y_test(i,1)=1;
    else
        Y_test(i,1)=0;
    end
end

% the performance of testing data %
y_validation=X_test * beta(2:end) + beta(1);
prob_validation=exp(y_validation)./(1 + exp(y_validation));
for i=1:n_test
    if prob_validation(i)>0.5
        Y_validation(i,1)=1;
    else
        Y_validation(i,1)=0;
    end
end
error_test=abs(Y_validation-Y_test);
error_number=length(find(nonzeros(error_test)))
beta_non_zero=length(nonzeros(beta))

%% Performance
[accurancy,sensitivity,specificity]=performance(Y_test,Y_validation);
fprintf('The accurancy of testing data (SCAD): %f\n' ,accurancy);
fprintf('The sensitivity of testing data (SCAD): %f\n' ,sensitivity);
fprintf('The specificity of testing data (SCAD): %f\n' ,specificity);


%% performance for training data
l1 = X0 * beta(2:end) + beta(1);
prob1=exp(l1)./(1 + exp(l1)); 
train_size=n;
for i=1:train_size
    if prob1(i)>0.5
        train_y(i)=1;
    else
        train_y(i)=0;
    end
end
Y0_origin(find(Y0_origin==-1))=0;

error_train=train_y'-Y0_origin;
error_number_train=length(nonzeros(error_train))

[accurancy_train,sensitivity_train,specificity_train]=performance(Y0_origin,train_y');
fprintf('The accurancy of training data(SCAD): %f\n' ,accurancy_train);
fprintf('The sensitivity of training data (SCAD): %f\n' ,sensitivity_train);
fprintf('The specificity of training data (SCAD): %f\n' ,specificity_train);

%% performance for beta
[accurancy_beta,sensitivity_beta,specificity_beta]=performance_beta(beta_true,beta);
fprintf('The accurancy of beta (SCAD): %f\n' ,accurancy_beta);
fprintf('The sensitivity of beta (SCAD): %f\n' ,sensitivity_beta);
fprintf('The specificity of beta (SCAD): %f\n' ,specificity_beta);