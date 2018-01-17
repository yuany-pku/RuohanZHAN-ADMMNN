function [l, truerate ] = Mnist_cla(batchdata,batchtargets,W1,W2,eps)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[~,N,numbatches] = size(batchdata);
m1 = size(W1,1);
m2 = size(W2,1);
A1 = zeros(m1,N,numbatches);
Z = zeros(m2,N,numbatches);
Y = batchtargets;
l = 0;

for j = 1:numbatches
A1(:,:,j) = h(W1*batchdata(:,:,j));
Z(:,:,j) = W2*A1(:,:,j);
l = l + sum(sum(max(1+Z(:,:,j)-repmat(sum(Y(:,:,j).*Z(:,:,j)),m2,1),0)));
end
l = l/(numbatches*N)-1 + eps/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));

[~,cla1] = max(Z);
[~,cla2] = max(batchtargets);
truerate = sum(cla1(:)==cla2(:))/(N*numbatches);

end

