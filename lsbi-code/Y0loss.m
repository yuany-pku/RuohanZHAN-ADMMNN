function [ Z ] = Y0loss( beta,q )
%UNTITLED9 Summary of this function goes here
%   solve: Z = argmin(max(0,1+Z)+\beta/2*(Z-q)^2)

% solve: Z = argmin(max(0,Z)+\beta/2*(Z-q)^2);
% t = 1/(2*beta);
% I1 = find(q<0);
% I2 = find(q>=t);
% Z = zeros(size(q));
% Z(I1) = q(I1);
% Z(I2) = q(I2)-t;

t = 1/beta;
Z = -1*ones(size(q));
I1 = find(q>t-1);
I2 = find(q<-1);
Z(I1) = q(I1) - t;
Z(I2) = q(I2);

end

