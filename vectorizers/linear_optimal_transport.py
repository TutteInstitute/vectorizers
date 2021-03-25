import numba
import numpy as np
from pynndescent.optimal_transport import (
    allocate_graph_structures,
    initialize_graph_structures,
    initialize_supply,
    initialize_cost,
    network_simplex_core,
    arc_id,
    ProblemStatus,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from pynndescent.distances import cosine, named_distances
from sklearn.utils.extmath import svd_flip, randomized_svd
from sklearn.preprocessing import normalize

import scipy.sparse

import os
import re
import tempfile


def str_to_bytes(size_str):
    parse_match = re.match(r"(\d+\.?\d*)([kMGT]?i?[Bb]?)$", size_str)
    if parse_match is None:
        raise ValueError(
            f"Invalid memory size string {size_str}; should be of the form '200M', '2G', etc."
        )

    if parse_match.group(2) in ("k", "kB", "kb"):
        return int(np.ceil(float(parse_match.group(1)) * 1024))
    elif parse_match.group(2) in ("M", "MB", "Mb"):
        return int(np.ceil(float(parse_match.group(1)) * 1024 ** 2))
    elif parse_match.group(2) in ("G", "GB", "Gb"):
        return int(np.ceil(float(parse_match.group(1)) * 1024 ** 3))
    elif parse_match.group(2) in ("T", "TB", "Tb"):
        return int(np.ceil(float(parse_match.group(1)) * 1024 ** 4))
    elif parse_match.group(2) in ("ki", "kiB", "kib"):
        return int(np.ceil(float(parse_match.group(1)) * 1000))
    elif parse_match.group(2) in ("Mi", "MiB", "Mib"):
        return int(np.ceil(float(parse_match.group(1)) * 1000 ** 2))
    elif parse_match.group(2) in ("Gi", "GiB", "Gib"):
        return int(np.ceil(float(parse_match.group(1)) * 1000 ** 3))
    elif parse_match.group(2) in ("Ti", "TiB", "Tib"):
        return int(np.ceil(float(parse_match.group(1)) * 1000 ** 4))
    else:
        return int(np.ceil(parse_match.group(1)))


@numba.njit(nogil=True, fastmath=True)
def project_to_sphere_tangent_space(euclidean_vectors, sphere_basepoints):
    result = np.zeros_like(euclidean_vectors)
    for i in range(result.shape[0]):
        unit_normal = sphere_basepoints[i] / np.sqrt(np.sum(sphere_basepoints[i] ** 2))
        scale = euclidean_vectors[i] @ unit_normal.astype(np.float64)
        result[i] = euclidean_vectors[i] - (scale * unit_normal)

    return result


@numba.njit(nogil=True, fastmath=True)
def tangent_vectors_scales(reference_vectors, image_vectors):
    result = np.zeros((reference_vectors.shape[0], 1), dtype=np.float32)
    for i in range(result.shape[0]):
        result[i, 0] = cosine(reference_vectors[i], image_vectors[i])
    return result


@numba.njit(nogil=True)
def get_transport_plan(flow, graph):
    n = graph.n
    m = graph.m
    result = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            arc = i * m + j
            flow_idx = arc_id(arc, graph)
            result[i, j] = flow[flow_idx]

    return result


@numba.njit()
def transport_plan(p, q, cost, max_iter=100000):
    node_arc_data, spanning_tree, graph = allocate_graph_structures(
        p.shape[0], q.shape[0], False,
    )
    initialize_supply(p, -q, graph, node_arc_data.supply)
    initialize_cost(cost, graph, node_arc_data.cost)

    init_status = initialize_graph_structures(graph, node_arc_data, spanning_tree)
    if init_status == False:
        raise ValueError(
            "Optimal transport inputs must be valid probability distributions."
        )
    solve_status = network_simplex_core(node_arc_data, spanning_tree, graph, max_iter,)
    if solve_status == ProblemStatus.INFEASIBLE:
        raise ValueError(
            "Optimal transport problem was INFEASIBLE. Please check " "inputs."
        )
    elif solve_status == ProblemStatus.UNBOUNDED:
        raise ValueError(
            "Optimal transport problem was UNBOUNDED. Please check " "inputs."
        )
    result = get_transport_plan(node_arc_data.flow, graph)

    return result


@numba.njit(nogil=True, parallel=True)
def chunked_pairwise_distance(data1, data2, dist=cosine, chunk_size=4):
    row_size = data1.shape[0]
    col_size = data2.shape[0]
    result = np.empty((row_size, col_size), dtype=np.float32)
    n_row_chunks = (row_size // chunk_size) + 1
    for chunk_idx in numba.prange(n_row_chunks):
        n = chunk_idx * chunk_size
        chunk_end_n = min((n + chunk_size), row_size)
        for m in range(0, col_size, chunk_size):
            chunk_end_m = min((m + chunk_size), col_size)
            for i in range(n, chunk_end_n):
                for j in range(m, chunk_end_m):
                    d = dist(data1[i], data2[j])
                    result[i, j] = d

    return result


# !! In place modification for efficiency
@numba.njit(nogil=True, fastmath=True)
def l2_normalize(vectors):
    for i in range(vectors.shape[0]):
        norm = 0.0
        for j in range(vectors.shape[1]):
            square = vectors[i, j] * vectors[i, j]
            norm += square

        norm = np.sqrt(norm)

        if norm > 0.0:
            for j in range(vectors.shape[1]):
                vectors[i, j] /= norm


@numba.njit(nogil=True, parallel=True)
def lot_vectors_sparse_internal(
    indptr,
    indices,
    data,
    sample_vectors,
    reference_vectors,
    reference_distribution,
    metric=cosine,
    max_distribution_size=256,
    chunk_size=256,
):
    n_rows = indptr.shape[0] - 1
    result = np.zeros((n_rows, reference_vectors.size), dtype=np.float64)
    n_chunks = (n_rows // chunk_size) + 1
    for n in numba.prange(n_chunks):
        chunk_start = n * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_rows)
        for i in range(chunk_start, chunk_end):
            row_indices = indices[indptr[i] : indptr[i + 1]]
            row_distribution = data[indptr[i] : indptr[i + 1]].astype(np.float64)

            if row_indices.shape[0] > max_distribution_size:
                best_indices = np.argsort(-row_distribution)[:max_distribution_size]
                row_indices = row_indices[best_indices]
                row_distribution = row_distribution[best_indices]

            row_sum = row_distribution.sum()

            if row_sum > 0.0:
                row_distribution /= row_sum

            row_vectors = sample_vectors[row_indices].astype(np.float64)

            if row_vectors.shape[0] > reference_vectors.shape[0]:
                cost = chunked_pairwise_distance(
                    row_vectors, reference_vectors, dist=metric
                )
            else:
                cost = chunked_pairwise_distance(
                    reference_vectors, row_vectors, dist=metric
                ).T

            current_transport_plan = transport_plan(
                row_distribution, reference_distribution, cost
            )
            transport_images = (
                current_transport_plan * (1.0 / reference_distribution)
            ).T @ row_vectors

            if metric == cosine:
                l2_normalize(transport_images)

            transport_vectors = transport_images - reference_vectors

            if metric == cosine:
                tangent_vectors = project_to_sphere_tangent_space(
                    transport_vectors, reference_vectors
                )
                l2_normalize(tangent_vectors)
                scaling = tangent_vectors_scales(transport_images, reference_vectors)
                transport_vectors = tangent_vectors * scaling

            result[i] = transport_vectors.flatten()

    return result


@numba.njit(nogil=True, parallel=True)
def lot_vectors_dense_internal(
    sample_vectors,
    sample_distributions,
    reference_vectors,
    reference_distribution,
    metric=cosine,
    max_distribution_size=256,
    chunk_size=256,
):
    n_rows = len(sample_vectors)
    result = np.zeros((n_rows, reference_vectors.size), dtype=np.float64)
    n_chunks = (n_rows // chunk_size) + 1
    for n in numba.prange(n_chunks):
        chunk_start = n * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_rows)
        for i in range(chunk_start, chunk_end):
            row_vectors = sample_vectors[i].astype(np.float64)
            row_distribution = sample_distributions[i]

            if row_vectors.shape[0] > max_distribution_size:
                best_indices = np.argsort(-row_distribution)[:max_distribution_size]
                row_vectors = row_vectors[best_indices]
                row_distribution = row_distribution[best_indices]

            row_sum = row_distribution.sum()

            if row_sum > 0.0:
                row_distribution /= row_sum

            if row_vectors.shape[0] > reference_vectors.shape[0]:
                cost = chunked_pairwise_distance(
                    row_vectors, reference_vectors, dist=metric
                )
            else:
                cost = chunked_pairwise_distance(
                    reference_vectors, row_vectors, dist=metric
                ).T

            current_transport_plan = transport_plan(
                row_distribution, reference_distribution, cost
            )
            transport_images = (
                current_transport_plan * (1.0 / reference_distribution)
            ).T @ row_vectors

            if metric == cosine:
                l2_normalize(transport_images)

            transport_vectors = transport_images - reference_vectors

            if metric == cosine:
                tangent_vectors = project_to_sphere_tangent_space(
                    transport_vectors, reference_vectors
                )
                l2_normalize(tangent_vectors)
                scaling = tangent_vectors_scales(transport_images, reference_vectors)
                transport_vectors = tangent_vectors * scaling

            result[i] = transport_vectors.flatten()

    return result


def lot_vectors_sparse(
    sample_vectors,
    weight_matrix,
    reference_vectors,
    reference_distribution,
    n_components=150,
    metric=cosine,
    random_state=None,
    max_distribution_size=256,
    block_size=16384,
):
    weight_matrix = weight_matrix.tocsr().astype(np.float64)
    weight_matrix = normalize(weight_matrix, norm="l1")
    if metric == cosine:
        sample_vectors = normalize(sample_vectors, norm="l2")

    n_rows = weight_matrix.indptr.shape[0] - 1
    n_blocks = (n_rows // block_size) + 1
    chunk_size = max(256, block_size // 64)

    if n_blocks == 1:
        lot_vectors = lot_vectors_sparse_internal(
            weight_matrix.indptr,
            weight_matrix.indices,
            weight_matrix.data,
            sample_vectors,
            reference_vectors,
            reference_distribution,
            metric=metric,
            max_distribution_size=max_distribution_size,
            chunk_size=chunk_size,
        )
        u, singular_values, v = randomized_svd(
            lot_vectors,
            n_components=n_components,
            n_iter=10,
            random_state=random_state,
        )
        result, components = svd_flip(u, v)

        return result, components

    singular_values = None
    components = None

    memmap_filename = os.path.join(tempfile.mkdtemp(), "lot_tmp_memmap.dat")
    saved_blocks = np.memmap(
        memmap_filename,
        mode="w+",
        shape=(n_rows, reference_vectors.size),
        dtype=np.float32,
    )

    for i in range(n_blocks):
        block_start = i * block_size
        block_end = min(n_rows, block_start + block_size)
        block = lot_vectors_sparse_internal(
            weight_matrix.indptr[block_start : block_end + 1],
            weight_matrix.indices,
            weight_matrix.data,
            sample_vectors,
            reference_vectors,
            reference_distribution,
            metric=metric,
            max_distribution_size=max_distribution_size,
            chunk_size=chunk_size,
        )

        if singular_values is not None:
            block_to_learn = np.vstack(
                (singular_values.reshape(-1, 1) * components, block)
            )
        else:
            block_to_learn = block

        u, singular_values, v = randomized_svd(
            block_to_learn,
            n_components=n_components,
            n_iter=10,
            random_state=random_state,
        )
        u, components = svd_flip(u, v)
        saved_blocks[block_start:block_end] = block

    saved_blocks.flush()
    del saved_blocks
    saved_blocks = np.memmap(
        memmap_filename,
        mode="r",
        shape=(n_rows, reference_vectors.size),
        dtype=np.float32,
    )
    result = saved_blocks @ components.T
    del saved_blocks
    os.remove(memmap_filename)

    return result, components


def lot_vectors_dense(
    sample_vectors,
    sample_distributions,
    reference_vectors,
    reference_distribution,
    n_components=150,
    metric=cosine,
    random_state=None,
    max_distribution_size=256,
    block_size=16384,
):
    if metric == cosine:
        sample_vectors = normalize(sample_vectors, norm="l2")

    n_rows = len(sample_vectors)
    n_blocks = (n_rows // block_size) + 1
    chunk_size = max(256, block_size // 64)

    if n_blocks == 1:
        lot_vectors = lot_vectors_dense_internal(
            sample_vectors,
            sample_distributions,
            reference_vectors,
            reference_distribution,
            metric=metric,
            max_distribution_size=max_distribution_size,
            chunk_size=chunk_size,
        )
        u, singular_values, v = randomized_svd(
            lot_vectors,
            n_components=n_components,
            n_iter=10,
            random_state=random_state,
        )
        result, components = svd_flip(u, v)

        return result, components

    singular_values = None
    components = None

    memmap_filename = os.path.join(tempfile.mkdtemp(), "lot_tmp_memmap.dat")
    saved_blocks = np.memmap(
        memmap_filename,
        mode="w+",
        shape=(n_rows, reference_vectors.size),
        dtype=np.float32,
    )

    for i in range(n_blocks):
        block_start = i * block_size
        block_end = min(n_rows, block_start + block_size)
        block = lot_vectors_dense_internal(
            sample_vectors[block_start:block_end],
            sample_distributions[block_start:block_end],
            reference_vectors,
            reference_distribution,
            metric=metric,
            max_distribution_size=max_distribution_size,
            chunk_size=chunk_size,
        )

        if singular_values is not None:
            block_to_learn = np.vstack(
                (singular_values.reshape(-1, 1) * components, block)
            )
        else:
            block_to_learn = block

        u, singular_values, v = randomized_svd(
            block_to_learn,
            n_components=n_components,
            n_iter=10,
            random_state=random_state,
        )
        u, components = svd_flip(u, v)
        saved_blocks[block_start:block_end] = block

    saved_blocks.flush()
    del saved_blocks
    saved_blocks = np.memmap(
        memmap_filename,
        mode="r",
        shape=(n_rows, reference_vectors.size),
        dtype=np.float32,
    )
    result = saved_blocks @ components.T
    del saved_blocks
    os.remove(memmap_filename)

    return result, components


class WassersteinVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=128,
        reference_size=None,
        reference_scale=0.01,
        metric="cosine",
        memory_size="2G",
        max_distribution_size=256,
        random_state=None,
    ):
        self.n_components = n_components
        self.reference_size = reference_size
        self.reference_scale = reference_scale
        self.metric = metric
        self.memory_size = memory_size
        self.max_distribution_size = max_distribution_size
        self.random_state = random_state

    def _get_metric(self):
        if type(self.metric) is str:
            if self.metric in named_distances:
                return named_distances[self.metric]
            else:
                raise ValueError(
                    f"Unsupported metric {self.metric} provided; metric should be one of {list(named_distances.keys())}"
                )
        elif callable(self.metric):
            return self.metric
        else:
            raise ValueError(
                f"Unsupported metric {self.metric} provided; metric should be a callable or one of {list(named_distances.keys())}"
            )

    def fit(self, X, y=None, vectors=None, **fit_params):
        if vectors is None:
            raise ValueError(
                "WassersteinVectorizer requires vector representations of points under the metric. "
                "Please pass these in to fit using the vectors keyword argument."
            )
        vectors = check_array(vectors)
        random_state = check_random_state(self.random_state)
        memory_size = str_to_bytes(self.memory_size)
        metric = self._get_metric()

        if scipy.sparse.isspmatrix(X) or type(X) is np.ndarray:
            if type(X) is np.ndarray:
                X = scipy.sparse.csr_matrix(X)

            if X.shape[1] != vectors.shape[0]:
                raise ValueError(
                    "distribution matrix must have as many columns as there are vectors"
                )

            X = normalize(X, norm="l1")

            if self.reference_size is None:
                reference_size = np.median(np.squeeze(np.array((X != 0).sum(axis=1))))
            else:
                reference_size = self.reference_size

            lot_dimension = reference_size * vectors.shape[1]
            block_size = memory_size // (lot_dimension * 8)
            u, s, v = scipy.sparse.linalg.svds(X)
            reference_center = v @ vectors
            if metric == cosine:
                reference_center /= np.sqrt(np.sum(reference_center ** 2))
            self.reference_vectors_ = reference_center + random_state.normal(
                scale=self.reference_scale, size=(reference_size, vectors.shape[1])
            )
            if metric == cosine:
                self.reference_vectors_ = normalize(self.reference_vectors_, norm="l2")

            self.reference_distribution_ = np.full(reference_size, 1.0 / reference_size)

            self.embedding_, self.components_ = lot_vectors_sparse(
                vectors,
                X,
                self.reference_vectors_,
                self.reference_distribution_,
                self.n_components,
                metric,
                random_state,
                self.max_distribution_size,
                block_size,
            )

        elif type(X) in ("list", "tuple", "numba.typed.List"):
            if self.reference_size is None:
                reference_size = np.median([len(x) for x in X])
            else:
                reference_size = self.reference_size

            distributions = numba.typed.List.empty_list(numba.float64[:])
            sample_vectors = numba.typed.List.empty_list(numba.float64[:, :])
            distributions.extend(tuple(X))
            sample_vectors.extend(tuple(vectors))

            lot_dimension = reference_size * vectors[0].shape[1]
            block_size = memory_size // (lot_dimension * 8)

            reference_center = np.mean(
                np.vstack(
                    [
                        X[i].reshape(-1, 1) * normalize(vectors[i], norm="l2")
                        for i in range(len(X))
                    ]
                ),
                axis=0,
            )
            if metric == cosine:
                reference_center /= np.sqrt(np.sum(reference_center ** 2))
            self.reference_vectors_ = reference_center + random_state.normal(
                scale=self.reference_scale, size=(reference_size, vectors.shape[1])
            )
            if metric == cosine:
                self.reference_vectors_ = normalize(self.reference_vectors_, norm="l2")

            self.reference_distribution_ = np.full(reference_size, 1.0 / reference_size)
            self.embedding_, self.components_ = lot_vectors_dense(
                sample_vectors,
                distributions,
                self.reference_vectors_,
                self.reference_distribution_,
                self.n_components,
                metric,
                random_state,
                self.max_distribution_size,
                block_size,
            )
        else:
            raise ValueError(
                "Input data not in a recognized format for WassersteinVectorizer"
            )

        return self

    def fit_transform(self, X, y=None, vectors=None, **fit_params):
        self.fit(X, vectors=vectors)
        return self.embedding_

    def transform(self, X, y=None, vectors=None, **transform_params):
        if vectors is None:
            raise ValueError(
                "WassersteinVectorizer requires vector representations of points under the metric. "
                "Please pass these in to transform using the vectors keyword argument."
            )
        vectors = check_array(vectors)
        memory_size = str_to_bytes(self.memory_size)
        metric = self._get_metric()

        if scipy.sparse.isspmatrix(X) or type(X) is np.ndarray:
            if type(X) is np.ndarray:
                X = scipy.sparse.csr_matrix(X)

            if X.shape[1] != vectors.shape[0]:
                raise ValueError(
                    "distribution matrix must have as many columns as there are vectors"
                )

            X = normalize(X, norm="l1")

            if metric == cosine:
                vectors = normalize(vectors, norm="l2")

            lot_dimension = self.reference_vectors_.size
            block_size = memory_size // (lot_dimension * 8)

            n_rows = X.indptr.shape[0] - 1
            n_blocks = (n_rows // block_size) + 1
            chunk_size = max(256, block_size // 64)

            result_blocks = []

            for i in range(n_blocks):
                block_start = i * block_size
                block_end = min(n_rows, block_start + block_size)
                block = lot_vectors_sparse_internal(
                    X.indptr[block_start : block_end + 1],
                    X.indices,
                    X.data,
                    vectors,
                    self.reference_vectors_,
                    self.reference_distribution_,
                    metric=metric,
                    max_distribution_size=self.max_distribution_size,
                    chunk_size=chunk_size,
                )

                result_blocks.append(block @ self.components_.T)

            return np.vstack(result_blocks)

        elif type(X) in ("list", "tuple", "numba.typed.List"):
            lot_dimension = self.reference_vectors_.size
            block_size = memory_size // (lot_dimension * 8)

            n_rows = len(X)
            n_blocks = (n_rows // block_size) + 1
            chunk_size = max(256, block_size // 64)

            distributions = numba.typed.List.empty_list(numba.float64[:])
            sample_vectors = numba.typed.List.empty_list(numba.float64[:, :])
            distributions.extend(tuple(X))
            sample_vectors.extend(tuple(vectors))

            result_blocks = []

            for i in range(n_blocks):
                block_start = i * block_size
                block_end = min(n_rows, block_start + block_size)
                block = lot_vectors_dense_internal(
                    sample_vectors[block_start:block_end],
                    distributions[block_start:block_end],
                    self.reference_vectors_,
                    self.reference_distribution_,
                    metric=metric,
                    max_distribution_size=self.max_distribution_size,
                    chunk_size=chunk_size,
                )

                result_blocks.append(block @ self.components_.T)

            return np.vstack(result_blocks)

        else:
            raise ValueError(
                "Input data not in a recognized format for WassersteinVectorizer"
            )
