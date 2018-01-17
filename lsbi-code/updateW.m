function [ Wp ] = updateW(w,waa,za,beta,kappa)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
% Wp = w-alpha*(kappa*beta)*(waa-za);
 Wp = w-(kappa*beta)*(waa-za);
end

