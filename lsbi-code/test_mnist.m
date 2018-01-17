%%% follow notations in Scaling Neural Networks with ADMM
% author: Ruohan Zhan
%% initialization

randn('seed',2016);
rand('seed',2016);

% gamma = 20*0.02;
% beta = 2*0.02;
gamma = 5*0.02;
beta = 20*0.02;
%kappa = 0.05;
% kappa = 0.001;
eps = 1.0e-8;
% alpha = 0.02;
Lnum1 = 300; % #nodes for layer one
% load traindata10;
% load testdata10;
% 
% % load traindata10;
% [m,N,numbatches] = size(batchdata);
% nu = N*numbatches*eps;
% [mt] = size(batchtargets,1);
% [~,N2,numbatches2] = size(testbatchtargets);
% data = [];
% for j = 1:numbatches
%     data = [data batchdata(:,:,j)];
% end
% meandata = mean(data,2);
% md = repmat(meandata,1,N,numbatches);
% stddata = std(data,0,2);
% sd = repmat(stddata,1,N,numbatches);
% % for j = 1:numbatches
% %     batchdata(:,:,j) = batchdata(:,:,j) - md;
% %     batchdata(:,:,j) = batchdata(:,:,j)./(sd+1.0e-6);
% % end
% batchdata = batchdata - md;
% batchdata = batchdata./(sd+1.0e-6);
% Y = batchtargets;
% 
% 
% 
% mdt = repmat(meandata,1,N2,numbatches2);
% sdt = repmat(stddata,1,N2,numbatches2);
% % for j = 1:numbatches2
% %     testbatchdata(:,:,j) = testbatchdata(:,:,j)-mdt;
% %     testbatchdata(:,:,j) = testbatchdata(:,:,j)./(sdt+1.0e-6);
% % end
% testbatchdata = testbatchdata - mdt;
% testbatchdata = testbatchdata./(sdt + 1.0e-6);

load traindata10 batchdata batchtargets
load testdata10 testbatchdata testbatchtargets
 [m,N,numbatches] = size(batchdata);
 [~,N2,numbatches2] = size(testbatchtargets);
 kappa = 1/(N*numbatches);
 
mt = size(batchtargets,1);

A0 = batchdata;
W1 = 0.01 *  randn(Lnum1,m);
W2 = 0.01 * randn(mt,Lnum1);
A1 = zeros(Lnum1,N,numbatches);
Z1 = zeros(Lnum1,N,numbatches);
Z2 = zeros(mt,N,numbatches);

for j = 1:numbatches
    Z1(:,:,j) = W1*A0(:,:,j);
    A1(:,:,j) = h(Z1(:,:,j));
    Z2(:,:,j) = W2*A1(:,:,j);
end


% Z2_train = Z2;
% Z2_train(Z2>0) = 1;
% Z2_train(Z2<=0) = -1;
% train_rate = [sum(sum(sum(Z2_train==Y)))/(N*numbatches)];
% test_rate = [];
% train_loss = [];
% test_loss = [];
% Z_test = zeros(1,N2,numbatches2);
% for j = 1:numbatches2
%     Z_test(:,:,j) = W2*h(W1*testbatchdata(:,:,j));
% end
% Z_test(Z_test>0) = 1;
% Z_test(Z_test<=0) = -1;
% 
% % test_rate = [test_rate Mnist_cla(testbatchdata,testbatchtargets,W1,W2)];
% test_rate = [test_rate sum(sum(sum(Z_test==testbatchtargets)))/(N2*numbatches2)];

[tl,tr] = Mnist_cla(batchdata,batchtargets,W1,W2,eps);
train_loss = [tl];
train_rate = [tr];
[~,ter] = Mnist_cla(testbatchdata,testbatchtargets,W1,W2,eps);
test_rate = [ter];



%% warm start
it_warm = 50;

% Nesterov accelarating

aa1 = zeros(m,m);
    for j = 1:numbatches
        aa1 = aa1+A0(:,:,j)*A0(:,:,j)'; 
    end
  
    
W1x = W1;
W2x = W2;
A1x = A1;
Z1x = Z1;
Z2x = Z2;
tic;
for i = 1:it_warm
    if i>1
        kappa = 1/(N*numbatches*i);
    end
   % update Z2
   tmp = Z2x;
for j = 1:numbatches
   Z2x(:,:,j) = MnistupdateZL(Y(:,:,j),1/kappa,Z2(:,:,j)-kappa*beta*(Z2(:,:,j)-W2*A1(:,:,j)));
end
Z2 = Z2x+(i-1)/(i+2)*(Z2x-tmp);

% update W2
tmp = W2x;
    za2 = zeros(mt,Lnum1);
    aa2 = zeros(Lnum1,Lnum1);
        for j = 1:numbatches
                    za2 = za2+Z2(:,:,j)*A1(:,:,j)';
                    aa2 = aa2+A1(:,:,j)*A1(:,:,j)';
        end
  W2x = updateW(W2,W2*aa2,za2,beta,kappa)-nu*W2;
  W2 = W2x+(i-1)/(i+2)*(W2x-tmp);
    
% update A1
tmp = A1x;
for j = 1:numbatches
   A1x(:,:,j) = updateA(A1(:,:,j),Z1(:,:,j),W2'*W2*A1(:,:,j),W2'*Z2(:,:,j),beta,gamma,kappa);
end
A1 = A1x+(i-1)/(i+2)*(A1x-tmp);

 % update Z1
 tmp = Z1x;
for j = 1:numbatches
   Z1x(:,:,j) = updateZ(Z1(:,:,j),A1(:,:,j),W1*A0(:,:,j),beta,kappa,gamma);
end
Z1 = Z1x+(i-1)/(i+2)*(Z1x-tmp);

% update W1
tmp = W1x;
    za1 = zeros(Lnum1,m);

    for j = 1:numbatches
        za1 = za1+Z1(:,:,j)*A0(:,:,j)';
    end
    W1x = updateW(W1,W1*aa1,za1,beta,kappa) - nu*W1;
W1 = W1x+(i-1)/(i+2)*(W1x-tmp);
 


   
% Z_train = zeros(1,N,numbatches);
% for j = 1:numbatches
%     Z_train(:,:,j) = W2*h(W1*A0(:,:,j));
% end
% Z_train(Z_train>0) = 1;
% Z_train(Z_train<=0) = -1;
% % train_rate = [train_rate Mnist_cla(batchdata,batchtargets,W1,W2)];
% train_rate = [train_rate sum(sum(sum(Z_train==Y)))/(N*numbatches)]
% train_loss = [train_loss hinge2(Z_train,Y)+1.0e-3/(2*N*numbatches)*(sum(sum(W1.^2))+sum(sum(W2.^2)))];
% 
% 
% Z_test = zeros(1,N2,numbatches2);
% for j = 1:numbatches2
%     Z_test(:,:,j) = W2*h(W1*testbatchdata(:,:,j));
% end
% Z_test(Z_test>0) = 1;
% Z_test(Z_test<=0) = -1;
% 
% % test_rate = [test_rate Mnist_cla(testbatchdata,testbatchtargets,W1,W2)];
% test_rate = [test_rate sum(sum(sum(Z_test==testbatchtargets)))/(N2*numbatches2)];
[tnl,tnct] = Mnist_cla(batchdata,batchtargets,W1,W2,eps);
[~,ttct] = Mnist_cla(testbatchdata,testbatchtargets,W1,W2,eps);
train_rate = [train_rate,tnct];
train_loss = [train_loss,tnl];
test_rate = [test_rate,ttct]


end

V = initialV(Z2,Y,kappa);
Vx = V;
W1x = W1;
W2x = W2;
A1x = A1;
Z1x = Z1;


kappa = 0.001/(N*numbatches);
%% train nn with admm
it_train = 100;
for i = 1:it_train
 % update V
 tmp = Vx;
 for j = 1:numbatches
     Vx(:,:,j) = updateV(V(:,:,j),Z2(:,:,j),W2*A1(:,:,j),beta );
 end
 V = Vx+(i-1)/(i+2)*(Vx-tmp);
 %V = Vx;
 
 % update Z2
 for j = 1:numbatches
   Z2 = MnistupdateZL(Y(:,:,j),1/kappa,V(:,:,j)*kappa);
 end
% update W2
    tmp = W2x;
    za2 = zeros(mt,Lnum1);
    aa2 = zeros(Lnum1,Lnum1);
        for j = 1:numbatches
                    za2 = za2+Z2(:,:,j)*A1(:,:,j)';
                    aa2 = aa2+A1(:,:,j)*A1(:,:,j)';
        end
    W2x = updateW(W2,W2*aa2,za2,beta,kappa)-nu*W2;
    W2 = W2x+(i-1)/(i+2)*(W2x-tmp);
  %  W2 = W2x;
  
 % update A1
 tmp = A1x;
for j = 1:numbatches
   A1x(:,:,j) = updateA(A1(:,:,j),Z1(:,:,j),W2'*W2*A1(:,:,j),W2'*Z2(:,:,j),beta,gamma,kappa);
end
A1 = A1x+(i-1)/(i+2)*(A1x-tmp);
%   A1 = A1x;
 % update Z1
tmp = Z1x;
for j = 1:numbatches
   Z1x(:,:,j) = updateZ(Z1(:,:,j),A1(:,:,j),W1*A0(:,:,j),beta,kappa,gamma);
end
   Z1 = Z1x + (i-1)/(i+2)*(Z1x-tmp);
 % Z1 = Z1x;
 
% update W1
    tmp = W1x;
    za1 = zeros(Lnum1,m);
    for j = 1:numbatches
        za1 = za1+Z1(:,:,j)*A0(:,:,j)';
    end
    W1x = updateW(W1,W1*aa1,za1,beta,kappa )-nu*W1;
    W1 = W1x + (i-1)/(i+2)*(W1x-tmp);
 %   W1 = W1x; 

  
% Z_train = zeros(1,N,numbatches);
% for j = 1:numbatches
%     Z_train(:,:,j) = W2*h(W1*A0(:,:,j));
% end
% Z_train(Z_train>0) = 1;
% Z_train(Z_train<=0) = -1;
% % train_rate = [train_rate Mnist_cla(batchdata,batchtargets,W1,W2)];
% train_rate = [train_rate sum(sum(sum(Z_train==Y)))/(N*numbatches)]
% train_loss = [train_loss hinge2(Z_train,Y)+1.0e-3/(2*N*numbatches)*(sum(sum(W1.^2))+sum(sum(W2.^2)))];
% 
% 
% Z_test = zeros(1,N2,numbatches2);
% for j = 1:numbatches2
%     Z_test(:,:,j) = W2*h(W1*testbatchdata(:,:,j));
% end
% Z_test(Z_test>0) = 1;
% Z_test(Z_test<=0) = -1;
% 
% % test_rate = [test_rate Mnist_cla(testbatchdata,testbatchtargets,W1,W2)];
% test_rate = [test_rate sum(sum(sum(Z_test==testbatchtargets)))/(N2*numbatches2)];
[tnl,tnct] = Mnist_cla(batchdata,batchtargets,W1,W2,eps);
[~,ttct] = Mnist_cla(testbatchdata,testbatchtargets,W1,W2,eps);
train_rate = [train_rate,tnct];
train_loss = [train_loss,tnl];
test_rate = [test_rate,ttct]


end
toc;
figure;
plot(train_rate);hold on;plot(test_rate);
% title('L-sbi: Mnist 0,1 accuracy');
title('L-sbi: Mnist accuracy');
legend('train data accuracy','test data accuracy');
hold off

figure;
plot(train_loss);
% title('L-sbi: Mnist 0,1 loss');
title('L-sbi: Mnist loss');


save learndata10  W1 W2 train_rate test_rate

