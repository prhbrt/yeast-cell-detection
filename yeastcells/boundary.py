import numpy
from .seamcarving import *

def calculate_path_features(coordinates, paths):
  non_circleness = []
  radii = []
  for coord, path in zip(coordinates, paths):
    if max(path) == 0:
      non_circleness.append(10)
      radii.append(0)
      continue
      
    x, y = polar_to_cartesial(coord[0], coord[1], path, delta=1/3)
    radii_ = numpy.sqrt((x - x.mean()) ** 2 +  (y - y.mean()) ** 2)
    radii.append(radii_.mean())
    non_circleness.append(radii_.var(ddof=0) / radii_.mean())
  return radii, non_circleness
