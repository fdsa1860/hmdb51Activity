% find histogram of dense trajectory

function h = findDenseHist(centers, samples, normalize)

nCenter = size(centers,2);
nSample = size(samples,2);

X2 = sum(centers.^2)' * ones(1,nSample);
Y2 = ones(nCenter,1) * sum(samples.^2);
XY = centers'*samples;
M = X2 - 2 * XY + Y2;
[~,ind2] = min(M);
h = hist(ind2, 1:nCenter);

if normalize
    h = h/sum(h);
end

end
