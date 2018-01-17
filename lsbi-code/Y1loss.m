function [ Z ] = Y1loss( beta,q )
%UNTITLED10 Summary of this function goes here
%  Solve: Z = argmin(max(0,1-Z)+\beta/2*(Z-q)^2)

t = 1/beta;
I1 = find(q<=1-t);
I2 = find(q>1);
Z = ones(size(q));
Z(I1) = q(I1)+t;
Z(I2) = q(I2);

end

