function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% 각 row 마다 모든 centroids 와의 거리를 구해서 최소의 거리를 가진 centroids 의 index 를 저장한다.
E = eye(K);
for i = 1:size(X,1)
	point = X(i, :);	
	% A = (point .- centroids);
	A = bsxfun(@minus, point, centroids);
	[value pos] = min(sum(A * A' .* E));
	idx(i) = pos;
end

% =============================================================

end

