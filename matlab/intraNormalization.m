function [X,intra] = intraNormalization(X,p)
% Input:
% X is D x N matrix, D is the dimension, N is the number of samples
% p is the optional vector. each element is the norm of samples around a
% cluster center
% Output:
% X is the normalized data
% intra is the normalization vector

N = size(X,2);

if nargin==1
    intra = sum(X.^2,2).^0.5;
elseif nargin == 2
    intra = p;
end

X = X./bsxfun(@times,intra,ones(1,N));

end