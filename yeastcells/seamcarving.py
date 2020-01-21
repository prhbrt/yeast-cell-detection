import numpy
from scipy.ndimage import geometric_transform


def seam_path(X, width=2):
    m, n = X.shape
    field = numpy.zeros((m, n + 2 * width))
    field[0, width:-width] = X[0]
    field[:, :width] = numpy.inf
    field[:, -width:] = numpy.inf
    d = field.itemsize

    for i in range(1, m):
        windows = numpy.ndarray(
            buffer=field[i-1].data,
            shape = (n, 2 * width + 1),
            strides = (d, d),
            dtype=field.dtype
        )

        field[i, width:-width] = X[i] + windows.min(1)
        field[i, width:-width] -= field[i, width:-width].min()

    path = numpy.zeros(m, numpy.uint32)
    path[-1] = numpy.argmin(field[-1])
    
    for i in reversed(range(m-1)):
        p = path[i+1]
        path[i] = p + numpy.argmin(field[i, p-width:p+width+1]) - width
        values = [x[p] if p < len(x) else numpy.nan for x, p in zip(X, path)]
    return field, path - width, values


def polar_to_cartesial(x, y, path, delta=1/3):
    angles = numpy.arange(len(path)) / len(path) * 2 * numpy.pi

    x = x + delta * path * numpy.cos(angles)
    y = y + delta * path * numpy.sin(angles)
    return x, y


def get_transform(x0, y0, n_theta, delta=1):
    def transform(coords):
        theta = 2.0 * numpy.pi * coords[0] / n_theta
        
        x = x0 + delta * coords[1] * numpy.cos(theta)
        y = y0 + delta * coords[1] * numpy.sin(theta)
        return y, x
    return transform


def get_ray(X_real, x, y, output_shape, delta):
    return geometric_transform(
        X_real,
        get_transform(x, y, output_shape[0], delta=delta),
        order = 3,
        mode = 'constant',
        cval= 2,
        prefilter = True,
        output_shape = output_shape
    )


def get_rays(X_real, coordinates, output_shape = (100, 100), delta=1/3):
    for i, (x, y, z) in enumerate(coordinates):
        yield get_ray(X_real[int(z), ...], x, y, output_shape, delta)
