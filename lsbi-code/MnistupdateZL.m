function  Z = MnistupdateZL(Y,beta,v)
%UNTITLED3 Summary of this function goes here
%   solve p+beta*z=beta*v
[m,n] = size(Y);
Z = zeros(m,n);
% A = zeros(m,m,m);
% b = zeros(m,m);
tmpI = eye(m);
% tmp1 = ones(m,1);
tmp11 = ones(m,m)/(2*m);
M = tmpI-tmp11/m;
% for j = 1:m
%     A(:,:,j) = tmpI-tmp1*tmpI(:,j)';
%     b(:,j) = tmp1-tmpI(:,j);
% end

    [~,loc] = max(Y); 
    for j = 1:m
    I = find(loc==j);
    if numel(I)>0
    Z(:,I) = solveSVM(j,beta,v(:,I),M);
    end
    end

end

