function [ z ] = Ty1( v,kappa )
%UNTITLED7 Summary of this function goes here
%  Solve y==1, p+1/kappa * z =v;


I1 = find(v>1/kappa);
I2 = find(v<1/kappa-1);
z = ones(size(v));
z(I1) = kappa*v(I1);
z(I2) = kappa*(v(I2)+1);

end

