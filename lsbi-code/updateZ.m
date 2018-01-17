function [Zp ] = updateZ( z,a,Wa,beta,kappa,gamma)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
 Zp = z-kappa*(gamma*gradh(z).*(h(z)-a)+beta*(z-Wa));
% Zp = z-kappa*(gamma*gradh(z).*(h(z)-a)+beta*(z-Wa));

end

