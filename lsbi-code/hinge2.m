function [ l ] = hinge2( Z,Y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[~,n,numbatches] = size(Z);
I1 = find(Y==1);
I2 = find(Y==-1);
l = sum(sum(sum(max(0,1-Z(I1)))));
l = l+sum(sum(sum(max(0,1+Z(I2)))));
l = l/(n*numbatches);

end

