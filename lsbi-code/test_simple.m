
%% initialization

figure;
randn('seed',2020);
rand('seed',2020);
gamma = 20 *0.02  ;
beta = 2 *0.02;
%kappa = 0.05;   % Damping factor
%alpha = 0.001;      % step size
kappa = 0.05; % large kappa has a higher risk of oscillation
% alpha = 0.02;
Lnum1 = 60; % #nodes for layer one
%load traindata;
X0 = rand([2,1000]);
X0tmp = 2*X0-1;
Y0 = X0tmp(1,:).*X0tmp(2,:); %double(sign(X0(1,:).*X0(2,:)));
%Y0 = double(sign(Y0+abs(Y0)));
Y0 = double(Y0>0);
Y0(Y0==0) = -1;

traindata = X0(:,1:800);

% centralize & normalize
trainmean = mean(traindata,2);
traindata = traindata - repmat(trainmean,1,800);
trainstd = std(traindata,0,2);
traindata = traindata./(repmat(trainstd,1,800));

traintarget = Y0(1:800);

testdata = X0(:,801:end);
testdata = testdata - repmat(trainmean,1,200);
testdata = testdata./repmat(trainstd,1,200);

testtarget = Y0(801:end);

[m,N] = size(traindata);
A0 = traindata;
Y = traintarget;

% num_warm = [200 500 1000 2000];
%num_warm = [10000];
num_warm = [1000];
TIME = [];
warmerrtrain = [];
warmerrtest = [];
slbierrtrain = [];
slbierrtest = [];
A02 = A0*A0';
W10 = 0.01*randn(Lnum1,2);
W20 = 0.01*randn(1,Lnum1);

for rr = 1:size(num_warm,2)

W1 = W10;
W2 = W20; 
Z1 = W1*A0;
A1 = h(Z1);
Z2 = W2*A1;
% A1 = randn(Lnum1,N);
% Z1 = randn(Lnum1,N);
% Z2 = randn(1,N);

%% warm start 
it_warm = num_warm(rr);
err_test = [];
err_train = [];
   Z_train = W2*h(W1*traindata);
      Z_train(Z_train>0) = 1;
   Z_train(Z_train<=0) = -1;
      err_train = [err_train, sum(Z_train~=traintarget)/800];
   
   Z_test = W2*h(W1*testdata);
   Z_test(Z_test>0) = 1;
Z_test(Z_test<=0) = -1;
%disp(['The iteration num is',num2str(i),'Test error is: '])
   err_test = [err_test, sum(Z_test~=testtarget)/200];
tic
for i = 1:it_warm
 % update W1 W2   
%       W1 = updateW(W1,W1*A02,Z1*A0',beta,kappa);
%       W2 = updateW(W2,W2*(A1*A1'),Z2*A1',beta,kappa);
% update Z1
%      Z1 = updateZ(Z1,A1,W1*A0,beta,kappa,gamma);
% update A1
%     A1 = updateA( A1,Z1,W2'*W2*A1,W2'*Z2,beta,ggamma,kappa );
  
     

% update Z2
  Z2 = updateZL_warmup(Y,kappa,Z2-kappa*beta*(Z2-W2*A1));
%     Z2 = LiupdateZL_warmup(Y,Z2,alpha*kappa,alpha*beta*kappa*(Z2-W2*A1));   
    W2 = updateW(W2,W2*(A1*A1'),Z2*A1',beta,kappa)-1.0e-5*W2;
 A1 = updateA( A1,Z1,W2'*W2*A1,W2'*Z2,beta,gamma,kappa);
    Z1 = updateZ(Z1,A1,W1*A0,beta,kappa,gamma);
    W1 = updateW(W1,W1*A02,Z1*A0',beta,kappa)-1.0e-5*W1;
    
   Z_train = W2*h(W1*traindata);
   Z_train(Z_train>0) = 1;
   Z_train(Z_train<=0) = -1;
   err_train = [err_train, sum(Z_train~=traintarget)/800];
   
   Z_test = W2*h(W1*testdata);
   Z_test(Z_test>0) = 1;
Z_test(Z_test<=0) = -1;
%disp(['The iteration num is',num2str(i),'Test error is: '])
   err_test = [err_test, sum(Z_test~=testtarget)/200];
end
V = initialV(Z2,Y,kappa);
warmerrtrain = [warmerrtrain,err_train(end)];
warmerrtest = [warmerrtest,err_test(end)];
%% train nn with admm
it_train = 1000;
err1 = zeros(1,it_train);
for i = 1:it_train
 % update W1 W2   
  

 
 % update V
   V = updateV(V,Z2,W2*A1,beta);
   
 % update Z2
   Z2 = updateZl(V,Y,kappa);
      W2 = updateW(W2,W2*(A1*A1'),Z2*A1',beta,kappa)-1.0e-5*W2;
      % update A1
   A1 = updateA( A1,Z1,W2'*W2*A1,W2'*Z2,beta,gamma,kappa);
   % update Z1
   Z1 = updateZ(Z1,A1,W1*A0,beta,kappa,gamma);
   
     W1 = updateW(W1,W1*A02,Z1*A0',beta,kappa)-1.0e-5*W1;
   
   A1_train = h(W1*traindata);
   Z_train = W2*A1_train;
   Z_train(Z_train>0) = 1;
   Z_train(Z_train<=0) = -1;
   err_train = [err_train, sum(Z_train~=traintarget)/800];
   
A1_test = h(W1*testdata);
Z_test = W2*A1_test;
Z_test(Z_test>0) = 1;
Z_test(Z_test<=0) = -1;
%disp(['The iteration num is',num2str(i),'Test error is: '])
err1(i) = sum(Z_test~=testtarget)/200;
end
err_test = [err_test err1];
%% test
%load testdata
plot(1:it_train+it_warm+1,err_train);
hold on
plot(1:it_train+it_warm+1,err_test);
hold on
TIME = [TIME toc]
slbierrtrain = [slbierrtrain,err_train(end)];
slbierrtest = [slbierrtest,err_test(end)];
end
% plot(1:it_train+it_warm,err_train);
% hold on
% plot(1:it_train+it_warm,err_test);
% hold on
%legend(sprintf('%dtrain',num_warm(1)),sprintf('%dtest',num_warm(1)),sprintf('%dtrain',num_warm(2)),sprintf('%dtest',num_warm(2)),'50train','50test','100train','100test')
% id=min(find(err_train<0.05));
% %plot(it_warm,err_train(it_warm),'x',id,err_train(id),'o');
% hold on
% id=min(find(err_test<0.05));
% %plot(it_warm,err_test(it_warm),'x',id,err_test(id),'o');
% hold on
% %legend('10','30','50','100')
legend('200warmup,err-train','200warmup,err-test','500warmup,err-train','500warmup,err-test','1000warmup,err-train','1000warmup,err-test','2000warmup,err-train','2000warmup,err-test')



