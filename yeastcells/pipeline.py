import time
import numpy
import sys

from .visualize import create_result_tiff
from .boundary import calculate_path_features
from .postprocessing import get_coordinates, inside_pilars_area
from .clustering import cluster_coordinates, cluster_len
from .seamcarving import get_rays, seam_path, polar_to_cartesial

from tqdm.autonotebook import tqdm as progressbar


class Result:
    def __init__(self, **kwargs):
        assert set(kwargs.keys()) == {
            'image',
            'y_pred', 'coordinates',
            'paths', 'cluster_labels',
            'coordinates_2', 'paths_2', 'cluster_labels_2'}
        for name, value in kwargs.items():
            setattr(self, name, value)

    def to_dict(self):
        cells = [list() for _ in range(max(self.cluster_labels_2) + 1)]

        for (x0, y0, z0), boundary, label in zip(self.coordinates_2, self.paths_2, self.cluster_labels_2):
            if label >= 0:
                cells[label].append({
                    'center': (x0, y0, z0),
                    'boundary': list(zip(*polar_to_cartesial(x0, y0, boundary, delta=1 / 3))),
                })
        return cells

    def to_tiff(self):
        return create_result_tiff(
            self.image, self.y_pred,
            self.coordinates, self.paths, self.cluster_labels,
            self.coordinates_2, self.paths_2, self.cluster_labels_2)


def pipeline(X,
    cell_localisation_model=None,
    false_positive_eliminator=None,
  ):
  print('predicting using U-NET...', file=sys.stderr)
  t0 = time.time()
  y_pred = cell_localisation_model.predict(
      X[:, ..., 0, None],
      batch_size=1,
      verbose=0
  )
  print(f'Took {time.time() - t0:8.02f}s', file=sys.stderr)
  t0 = time.time()
  print('Determining cell centers...', file=sys.stderr)

  coordinates = numpy.array(list(get_coordinates(X, y_pred)))
  print(f'Took {time.time() - t0:8.02f}s', file=sys.stderr)
  t0 = time.time()

  if len(coordinates) == 0:
      print(f'WARNING: no cells found')
      return Result(
          image=X,
          y_pred=y_pred,
          coordinates=coordinates,
          paths=[],
          cluster_labels=[],
          coordinates_2=[],
          paths_2 = numpy.zeros((0, 4)),
          cluster_labels_2=[]
      )
  
  print('Clustering coordinates across frames...', file=sys.stderr)
  cluster_labels = cluster_coordinates(coordinates)
  cluster_lengths = cluster_len(cluster_labels, if_negative=-100)

  X_for_rays = X[..., 0] + (1.0 - inside_pilars_area(X))
  print(f'Took {time.time() - t0:8.02f}s', file=sys.stderr)
  t0 = time.time()

  print("Finding cell boundaries using seam carving, this takes many minutes...", file=sys.stderr)
  paths = [
      seam_path(ray, width = 5)[1:]
      for ray in get_rays(X_for_rays, progressbar(coordinates, file=sys.stderr))
  ]

  paths = numpy.array([p for p, _ in paths])

  distances = [
    numpy.sqrt((xx[0] - xx[-1]) ** 2 + (yy[0] - yy[-1]) ** 2)
    for path, (x, y, z_) in zip(paths, coordinates)
    for xx, yy in [polar_to_cartesial(x, y, path, delta=1/3)] # alias
  ]
  print(f'Took {time.time() - t0:8.02f}s', file=sys.stderr)
  t0 = time.time()

  print("Finding features for the path, including the average radius and how "
        "non-circular it is...", file=sys.stderr)
  radii, non_circleness = calculate_path_features(coordinates, paths)
  print(f'Took {time.time() - t0:8.02f}s', file=sys.stderr)
  t0 = time.time()

  print("Finding false positives...", file=sys.stderr)
  eliminated_false_positives = false_positive_eliminator.predict_proba(
    list(zip(
        distances,
        radii,
        non_circleness,
        cluster_labels >= 0,
        cluster_lengths
    ))
  )[..., 1] >= .55

  coordinates_2 = coordinates[eliminated_false_positives]
  paths_2 = [paths[i] for i, t in enumerate(eliminated_false_positives) if t]
  cluster_labels_2 = cluster_coordinates(coordinates_2)

  return Result(
      image=X,
      y_pred=y_pred,
      coordinates=coordinates,
      paths=paths,
      cluster_labels=cluster_labels,
      coordinates_2=coordinates_2,
      paths_2 = numpy.array(paths_2),
      cluster_labels_2=cluster_labels_2
  )
