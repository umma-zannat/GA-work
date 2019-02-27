import cython

import numpy
cimport numpy
from libc.math cimport isnan

ctypedef numpy.uint8_t uint8

def evaluate(double[:, :] query_points, int[:] closest_simplex, double[:, :, :] transform, int[:, :] simplices, int[:, :] neighbors, double[:] values):
    cdef double[:, ::1] query_points_c
    cdef int[::1] closest_simplex_c
    cdef double[:, :, ::1] transform_c
    cdef int[:, ::1] simplices_c
    cdef int[:, ::1] neighbors_c
    cdef double[::1] values_c

    cdef uint8[::1] visited
    cdef int[::1] queue
    cdef double[::1] c
    cdef double[::1] result

    cdef int nquery
    cdef int ndim
    cdef int nsimplex

    nquery = query_points.shape[0]
    ndim = query_points.shape[1]
    nsimplex = simplices.shape[0]

    query_points_c = numpy.ascontiguousarray(query_points)
    closest_simplex_c = numpy.ascontiguousarray(closest_simplex)
    transform_c = numpy.ascontiguousarray(transform)
    simplices_c = numpy.ascontiguousarray(simplices)
    neighbors_c = numpy.ascontiguousarray(neighbors)
    values_c = numpy.ascontiguousarray(values)

    visited = numpy.full((nsimplex,), fill_value=0, dtype=numpy.uint8)
    queue = numpy.full((nsimplex,), fill_value=-1, dtype=numpy.intc)

    result = numpy.full((nquery,), fill_value=numpy.nan, dtype=numpy.double)

    c = numpy.zeros((ndim + 1,))

    evaluate_cython(query_points_c, closest_simplex_c, transform_c, simplices_c, neighbors_c, values_c, visited, queue, c, result)

    return numpy.asarray(result)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void evaluate_cython(double[:, ::1] query_points, int[::1] closest_simplex, double[:, :, ::1] transform,
                          int[:, ::1] simplices, int[:, ::1] neighbors, double[::1] values,
                          uint8[::1] visited, int[::1] queue, double[::1] c, double[::1] result) nogil:
    cdef int iquery
    cdef int nquery

    nquery = query_points.shape[0]

    for iquery in xrange(nquery):
        evaluate_single(iquery, query_points, closest_simplex[iquery], transform, simplices, neighbors, values, visited, queue, c, result)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int enqueue(int[::1] queue, int qhead, int qtail, int n) nogil:
    cdef int i

    for i in xrange(qhead, qtail):
        if queue[i] == n:
            return 0

    queue[qtail] = n
    return 1


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void barycentric_coordinates(int iquery, double[:, ::1] query_points, int simplex, double[:, :, ::1] transform, double[::1] c) nogil:
    cdef int ndim
    cdef int i, j

    ndim = query_points.shape[1]

    c[ndim] = 1.0
    for i in xrange(ndim):
        c[i] = 0.0

        for j in xrange(ndim):
            c[i] += transform[simplex, i, j] * (query_points[iquery, j] - transform[simplex, ndim, j])

        c[ndim] -= c[i]


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void evaluate_single(int iquery, double[:, ::1] query_points, int simplex, double[:, :, ::1] transform,
                          int[:, ::1] simplices, int[:, ::1] neighbors, double[::1] values,
                          uint8[::1] visited, int[::1] queue, double[::1] c, double[::1] result) nogil:
    cdef int ndim
    cdef int nsimplex
    cdef int i
    cdef int n
    cdef int qhead
    cdef int qtail
    cdef uint8 failed
    cdef int visited_min
    cdef int visited_max

    ndim = query_points.shape[1]
    nsimplex = visited.shape[0]

    visited_min = simplex
    visited_max = simplex

    qhead = 0
    qtail = 0

    while True:
        failed = 0
        visited[simplex] = 1

        if simplex < visited_min:
            visited_min = simplex
        if simplex > visited_max:
            visited_max = simplex

        barycentric_coordinates(iquery, query_points, simplex, transform, c)

        if isnan(c[ndim]):
            failed = 1

            for i in xrange(ndim + 1):
                n = neighbors[simplex, i]

                if n != -1 and visited[n] == 0:
                    qtail = qtail + enqueue(queue, qhead, qtail, n)

        else:
            for i in xrange(ndim + 1):
                if c[i] < 0.:
                    failed = 1
                    n = neighbors[simplex, i]

                    if n != -1 and visited[n] == 0:
                        qtail = qtail + enqueue(queue, qhead, qtail, n)

        if not failed:
            # found the containing simplex
            result[iquery] = 0.0

            for i in xrange(ndim + 1):
                result[iquery] += c[i] * values[simplices[simplex, i]]

            # cleanup visited array
            for i in xrange(visited_min, visited_max + 1):
                visited[i] = 0

            return

        else:
            if qhead >= qtail:
                # empty queue so complete failure

                # cleanup visited array
                for i in xrange(visited_min, visited_max + 1):
                    visited[i] = 0
                return

            # try next simplex in queue
            simplex = queue[qhead]
            qhead = qhead + 1
