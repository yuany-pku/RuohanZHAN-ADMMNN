function [ z ] = Ty0( v,kappa)
%UNTITLED8 Summary of this function goes here
%   Solve y==-1, p+1/kappa * z =v;

% z = zeros(size(v));
% I1 = find(v<0);
% I2 = find(v>1);
% z(I1) = kappa*v(I1);
% z(I2) = kappa*(v(I2)-1);

z = -1 * ones(size(v));
I1 = find(v<-1/kappa);
I2 = find(v>1-1/kappa);
z(I1) = kappa*v(I1);
z(I2) = kappa*(v(I2)-1);

end

