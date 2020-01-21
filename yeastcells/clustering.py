import numpy
from scipy.ndimage import geometric_transform
from scipy.sparse import dok_matrix
from sklearn.cluster import DBSCAN
from collections import Counter


def cluster_len(cluster_labels, if_negative=-100):
  counts = Counter(cluster_labels)
  return [
      if_negative if i < 0 else counts[i]
      for i in cluster_labels
  ]


def get_distances(coordinates, max_dz=2):
  """For clustering we use a precomputed distance matrix. Since if the
  z-coordinates (time) differ a lot between two cells (say > 2), then
  we assume this is not the same, or that it should be connect by coordinates
  in the frames in between. Hence we only compute distances where dz > 0 and
  <= `max_dz` (e.g. <= `2`). For `max_dz==2`, this results in approximately 2%
  of the the matrix entries being non-zero, and hence is a memory relief
  when stored sparsely."""

  z = numpy.array([-numpy.inf] + list(coordinates[:, 2]) + [numpy.inf])
  boundaries = numpy.where(z[:-1] != z[1:])[0]
  boundaries = list(zip(boundaries[:-1], boundaries[1:]))

  distances = dok_matrix((len(coordinates), len(coordinates)), dtype=numpy.float64)

  for (a0, a1), ahead in zip(boundaries, zip(*(
      boundaries[i+1:] for i in range(max_dz)
  ))):
    for b0, b1 in ahead:
      d = ((
          coordinates[a0:a1, None, :2] - 
          coordinates[None, b0:b1, :2]
      ) ** 2).sum(2)
      distances[a0:a1, b0:b1] = d
      distances[b0:b1, a0:a1] = d.T
  return distances.tocsr()


def cluster_coordinates(coordinates, eps=30, min_samples=5, max_dz=4):
  precomputed = get_distances(coordinates, max_dz=max_dz)
  clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
  clusters = clustering.fit(precomputed)
  
  return clustering.labels_
