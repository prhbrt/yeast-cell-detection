import numpy
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.measurements import label
from skimage.morphology import dilation, erosion
from collections import Counter


def filter_cc(im, minsize=50):
    """Creates a new binary image from `im`, also a binary image,
    with only the connected components left larger than `minsize`"""
    labels, nlabels = label(im)
    for i in range(1, nlabels+1):
        mask = labels == i
        labels[mask] = 0 if mask.sum() < minsize else 1
    return labels


def inside_pilars_area(X):
    """Returns a 2D binary array, True where the pilars are and False in other areas"""
    average = X[..., 0].mean(0)
    low = dilation(filter_cc(average < 0.15, 200), numpy.ones((20, 20)))
    high = dilation(filter_cc(average > 0.6, 200), numpy.ones((20, 20)))

    boundaries = low * high

    pillared = label(True - boundaries)[0]

    largest_cc = max(Counter(pillared.ravel()).items(), key=lambda x: x[1])[0]

    pillared = (pillared != 0) * (pillared != largest_cc)
    return pillared


def connected_component_coordinates(X, y_pred, selector=lambda values: True):
    """Yields centers of the connected components in `y_pred`, `X` is the same shape as y_pred
    with the coresponding (continuous, floating point) pixel-values. If specified, `selector`
    is called on a list of the corresponding gray values of the connected component, and the
    coordinate is only returned if the call returns True."""
    coordinates = []
    labels, n_labels = label(y_pred)

    for l in range(1, n_labels + 1):
        y, x = numpy.where(labels == l)
        values = [X[y__, x__, 0] for x__, y__ in zip(x, y)]

        if selector(values):
            yield (x.mean(), y.mean())
    return coordinates


def overlay(X, y_pred, y_true=None):
    merged = X[..., None].repeat(3, axis=2)
    merged[..., 0] = numpy.maximum(merged[..., 0], y_pred)
    if y_true is not None:
        merged[..., 2] = numpy.maximum(merged[..., 0], y_true)
    return merged


def get_coordinates(X, y_pred):
  pillared = inside_pilars_area(X)
  
  for z, (X_, y_) in enumerate(zip(X, y_pred)):
    for coord in connected_component_coordinates(
        X_, pillared * (y_[..., 0] / y_[..., 0].max() ** 16 > 0.5),
        selector = lambda values: numpy.mean([
            x < 0.15 or x > 0.7
            for x in values
        ]) <= 0.6
    ):
      yield coord + (z,)
