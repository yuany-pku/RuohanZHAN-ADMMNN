function [ A] = updateA( a,z,WWa,Wz,beta,gamma,kappa )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
A = a-kappa*(gamma*(a-h(z))+beta*(WWa-Wz));

end

