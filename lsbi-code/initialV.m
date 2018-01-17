function [ V ] = initialV( Z,Y,kappa)
%UNTITLED12 Summary of this function goes here
%   Detailed explanation goes here
V = Z./kappa;

% I0 = find(Y==0);
I0 = find(Y==-1);
I1 = find(Y==1);
V(I0) = iniLam0(Z(I0))+V(I0);
V(I1) = iniLam1(Z(I1))+V(I1);

end

