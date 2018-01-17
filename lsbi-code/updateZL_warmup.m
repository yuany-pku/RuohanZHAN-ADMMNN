function [ Z] = updateZL_warmup( Y,delta,q )
%UNTITLED7 Summary of this function goes here
% loss function, seperatable, piecewise linear
I1 = find(Y==1);
% I0 = find(Y==0);
I0 = find(Y==-1);

Z = zeros(size(Y));
beta = 1/(delta);
Z(I0) = Y0loss(beta,q(I0));
Z(I1) = Y1loss(beta,q(I1));

end

