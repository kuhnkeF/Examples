# Given a set of 2-dimensional observations (points)
# for each point, find another point in the set with the smallest Euclidean distance.

import numpy as np
from scipy.spatial.distance import pdist, squareform

#                     x  y
points = np.asarray([[0, 0],
                     [1, 2],
                     [3, 3],
                     [0, 9]])  # matrix of points with x y coordinates (N by 2, with N observations)

N = points.shape[0]  # we have N = 4 observations/points

# compute matrix of all pairwise Euclidean distances
ps = squareform(pdist(points))

# for each point i, find the distance to the nearest point (but omit the distance to itself)
for i in range(N):
    ps[i, i] = np.inf  # set the distance of the point to itself to infinity
    minimal_distance = np.min(ps[i, :])
    nearest_point = np.argmin(ps[i, :])

    # example output
    print(points[i], 'has the smallest distance to', points[nearest_point], 'the distance is', minimal_distance)

# output of this program is
# [0 0] has the smallest distance to [1 2] the distance is 2.23606797749979
# [1 2] has the smallest distance to [0 0] the distance is 2.23606797749979
# [3 3] has the smallest distance to [1 2] the distance is 2.23606797749979
# [0 9] has the smallest distance to [3 3] the distance is 6.708203932499369
