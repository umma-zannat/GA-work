import numpy
from numpy import linalg

from scipy.spatial import ConvexHull, Delaunay, cKDTree as KDTree
from scipy.interpolate import griddata, LinearNDInterpolator

from .utils import timing

from .core import evaluate


def ndlinear(points, values, query_points, fill_value=numpy.nan):
    npoints, ndim = points.shape
    nquery, _ = query_points.shape

    assert ndim == _

    with timing('kdtree'):
        tree = KDTree(points)
        distances, closest_vertex = tree.query(query_points)

    with timing('delaunay'):
        delaunay = Delaunay(points)

    with timing('ancillary'):
        vertex_to_simplex = delaunay.vertex_to_simplex
        closest_simplex = vertex_to_simplex[closest_vertex]
        assert numpy.all(closest_simplex != -1)
        neighbors = delaunay.neighbors
        simplices = delaunay.simplices

    with timing('transform'):
        transform = delaunay.transform

    with timing('directed search'):
        return evaluate(query_points, closest_simplex, transform, simplices, neighbors, values)
