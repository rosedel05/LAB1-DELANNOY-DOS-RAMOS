# cython: boundscheck=False, wraparound=False, initializedcheck=False, language_level=3
import numpy as np
cimport numpy as np


def knn_predict(double[:, ::1] x_train,
                int[:] y_train,
                double[:, ::1] x_query,
                int k,
                int n_classes):
    """Compute K-NN predictions with a plain C loop implementation."""
    cdef Py_ssize_t n_query = x_query.shape[0]
    cdef Py_ssize_t n_train = x_train.shape[0]
    cdef Py_ssize_t n_features = x_train.shape[1]
    cdef Py_ssize_t q, j, f, i, c_idx, worst_pos
    cdef double dist, diff, worst_dist
    cdef int best_vote

    if n_train == 0:
        raise ValueError("Le jeu d'entrainement ne doit pas être vide.")
    if k < 1:
        raise ValueError("Le nombre de voisins doit etre positif.")
    if k > n_train:
        k = n_train
    if n_classes < 1:
        raise ValueError("Il faut au moins une classe pour voter.")

    for j in range(n_train):
        if y_train[j] < 0 or y_train[j] >= n_classes:
            raise ValueError("Les labels doivent etre codes entre 0 et n_classes-1.")

    result = np.empty(n_query, dtype=np.intc)
    best_distances = np.empty(k, dtype=np.float64)
    best_indices = np.empty(k, dtype=np.intc)
    votes = np.zeros(n_classes, dtype=np.intc)

    cdef int[:] res = result
    cdef double[:] best_dist = best_distances
    cdef int[:] best_idx = best_indices
    cdef int[:] vote_counts = votes

    for q in range(n_query):
        for i in range(k):
            best_dist[i] = 1e308
            best_idx[i] = -1

        for j in range(n_train):
            dist = 0.0
            for f in range(n_features):
                diff = x_query[q, f] - x_train[j, f]
                dist += diff * diff

            worst_pos = 0
            worst_dist = best_dist[0]
            for i in range(1, k):
                if best_dist[i] > worst_dist:
                    worst_dist = best_dist[i]
                    worst_pos = i

            if dist < worst_dist:
                best_dist[worst_pos] = dist
                best_idx[worst_pos] = j

        for c_idx in range(n_classes):
            vote_counts[c_idx] = 0

        for i in range(k):
            if best_idx[i] >= 0:
                vote_counts[y_train[best_idx[i]]] += 1

        res[q] = 0
        best_vote = vote_counts[0]
        for c_idx in range(1, n_classes):
            if vote_counts[c_idx] > best_vote:
                best_vote = vote_counts[c_idx]
                res[q] = c_idx

    return result
