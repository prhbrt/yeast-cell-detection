import cv2
import numpy
from yeastcells.seamcarving import polar_to_cartesial
# from .pipeline import pipeline


def draw_centers_and_contours(shape, coordinates, paths, cluster_labels):
  for z in range(int(coordinates[:, 2].max() + 1)):
    indices = numpy.where(coordinates[:, 2] == z)[0]
    contour_buffer = numpy.zeros(shape, dtype=numpy.uint16)
    center_buffer = numpy.zeros(shape, dtype=numpy.uint16)

    coordinates[indices]
    cluster_labels[indices]
    paths[indices]
    for path, (x, y, z), label in zip(paths[indices], coordinates[indices], cluster_labels[indices]):
      pixel_value = 65535 - int(label)
      xx, yy = polar_to_cartesial(x, y, path, delta=1/3)
      x_, y_ = numpy.round([x, y]).astype(numpy.int32)
      cv2.circle(center_buffer, (x_, y_), 3, pixel_value, -1)
      
      cv2.polylines(
          contour_buffer,
          numpy.array([list(zip(xx, yy))]).astype(numpy.int32),
          True, pixel_value,
          thickness=1,
          lineType=8,
          shift=0
      )
    yield int(z), center_buffer, contour_buffer


def create_result_tiff(X, y_pred, coordinates, paths, cluster_labels,
                       coordinates_2, paths_2, cluster_labels_2):
  result = numpy.zeros(X.shape[:3] + (6,), dtype=numpy.uint16)
  result[..., 0] = (X[..., 0] * 65535).astype(numpy.uint16)
  result[..., 1] = (y_pred[..., 0] * 65535).astype(numpy.uint16)

  for z, centers, contours in draw_centers_and_contours(
      result.shape[1:3], coordinates, paths, cluster_labels):
    result[z, ..., 2], result[z, ..., 3] = centers, contours

  for z, centers, contours in draw_centers_and_contours(
      result.shape[1:3], coordinates_2, paths_2, cluster_labels_2):
    result[z, ..., 4], result[z, ..., 5] = centers, contours
  return result
#
# def track_cells_as_tiff(X, cell_localisation_model, false_positive_eliminator):
#   (y_pred,
#    coordinates,
#    paths,
#    cluster_labels,
#    coordinates_2,
#    paths_2,
#    cluster_labels_2) = pipeline(X, cell_localisation_model, false_positive_eliminator)
#
#   return create_result_tiff(
#     X, y_pred,
#     coordinates, paths, cluster_labels,
#     coordinates_2, paths_2, cluster_labels_2)
