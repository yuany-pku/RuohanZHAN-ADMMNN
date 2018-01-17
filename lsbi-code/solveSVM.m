function    z = solveSVM(j,beta,v,M)
%UNTITLED5 Summary of this function goes here
%   class j,solve p+beta*z=v
[m,n] = size(v);
ej = zeros(m,1);
ej(j) = 1;
c = beta*(v-repmat(ej,1,n));
Alam = M*c;
I = [1:j-1 j+1:m];
Alam(I) = max(Alam(I),0);
Alam(I) = min(Alam(I),1);
Alam(j) = -sum(Alam(I));
z = v-Alam/beta;
end

