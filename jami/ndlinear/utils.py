from contextlib import contextmanager
from timeit import default_timer as timer
import sys

import numpy
from scipy.interpolate import LinearNDInterpolator


def echo(msg, *args, **kwargs):
    print(msg.format(*args, **kwargs))
    sys.stdout.flush()


@contextmanager
def timing(msg='elapsed'):
    start_time = timer()
    yield
    echo("{}: {:.3f}s", msg, timer() - start_time)


def ndlinear_orig(points, values, query_points, fill_value=numpy.nan, rescale=False):
    with timing('delaunay'):
        ip = LinearNDInterpolator(points, values, fill_value=fill_value, rescale=rescale)

    with timing('evaluation'):
        return ip(query_points)
