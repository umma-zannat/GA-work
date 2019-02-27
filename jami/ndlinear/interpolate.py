import numpy
from numpy import linalg

from scipy.spatial import ConvexHull, Delaunay, cKDTree as KDTree
from scipy.interpolate import griddata, LinearNDInterpolator

from .utils import timing


def directed(qp, simplex, transform, simplices, neighbors, values):
    [ndim,] = qp.shape

    visited = set()
    queue = []

    c = numpy.zeros((ndim + 1,))

    while True:
        d = numpy.matmul(transform[simplex, :-1, :], qp - transform[simplex, -1, :])
        d_sum = d.sum()
        c[:-1] = d
        c[-1] = 1. - d_sum

        if numpy.any(c < 0.) or numpy.isnan(d_sum):
            visited.add(simplex)
            candidates = [neighbors[simplex, k] for k in range(ndim + 1) if c[k] < 0. or numpy.isnan(c[k])]
            queue.extend([n for n in candidates if n != -1 and n not in visited and n not in queue])

            if queue == []:
                return numpy.nan
            else:
                simplex = queue.pop(0)
        else:
            return numpy.dot(c, values[simplices[simplex]])


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
            result = numpy.full((nquery,), numpy.nan, dtype=values.dtype)

            for i in range(nquery):
                result[i] = directed(query_points[i], closest_simplex[i], transform, simplices, neighbors, values)

        return result
