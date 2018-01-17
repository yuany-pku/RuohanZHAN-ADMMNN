function [ z ] = updateZl(v, y,kappa )
%UNTITLED10 Summary of this function goes here
%   Sovle p+1/kappa * z =v

I0 = find(y==-1);
I1 = find(y==1);
z = zeros(size(v));
z(I0) = Ty0(v(I0),kappa);
z(I1) = Ty1(v(I1),kappa);
end

