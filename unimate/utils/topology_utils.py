"""Skeleton topology utilities.

All public functions accept a parent index array (J,) as list or numpy array
where ``parents[j]`` is the parent of joint *j* (root has parent -1 or itself).

Sections:
    1. Internal helpers
    2. Graph representations — edge index, joint depths
    3. Graph positional encodings — Laplacian eigenvectors
    4. Pairwise relations — edge types, topology distances
"""

from collections import deque

import numpy as np
import torch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


# ---------------------------------------------------------------------------
# 1. Internal helpers
# ---------------------------------------------------------------------------

def _parse_parents(parents):
    """Normalize *parents* to a 1-D int64 numpy array."""
    return np.asarray(parents, dtype=np.int64).ravel()


def _children_map(parents):
    """Build a list-of-lists mapping each joint to its children.

    Args:
        parents: List of parent indices (J,).

    Returns:
        List of lists, where ``children[j]`` contains the children of joint *j*.
    """
    n = len(parents)
    children = [[] for _ in range(n)]
    for c, p in enumerate(parents):
        if p is None or p == -1:
            continue
        if not (0 <= p < n):
            raise ValueError(f"Invalid parent index parents[{c}]={p}")
        children[p].append(c)
    return children


def _adjacency_list(parents):
    """Build an undirected adjacency list from *parents*.

    Args:
        parents: List of parent indices (J,).

    Returns:
        List of lists, where ``adj[j]`` contains the neighbors of joint *j*.
    """
    n = len(parents)
    adj = [[] for _ in range(n)]
    for c, p in enumerate(parents):
        if p is None or p == -1:
            continue
        adj[c].append(p)
        adj[p].append(c)
    return adj



# ---------------------------------------------------------------------------
# 2. Graph representations
# ---------------------------------------------------------------------------

def compute_edge_indexs(parents):
    """Build a PyG-style edge index array from a skeleton parent array.

    Undirected: includes both parent->child and child->parent edges.
    Self-loops are excluded.

    Args:
        parents: Parent indices (J,).

    Returns:
        ``np.ndarray`` of shape ``[2, 2*(J-1)]`` with dtype int64.
    """
    parents = _parse_parents(parents)
    children = np.arange(len(parents))
    valid = (children != parents) & (parents >= 0)
    p = parents[valid]
    c = children[valid]
    return np.stack([np.concatenate([p, c]), np.concatenate([c, p])], axis=0)


def compute_joint_depths(parents):
    """Compute the depth of each joint in the kinematic tree.

    Handles arbitrary joint ordering (parent index may be > child index)
    by iterating until all depths are resolved.

    Args:
        parents: Parent indices (J,).

    Returns:
        ``np.ndarray`` of shape ``(J,)`` with dtype int64 (root=0).
    """
    parents = list(parents)
    n = len(parents)
    depths = np.full(n, -1, dtype=np.int64)

    for j in range(n):
        if parents[j] == -1 or parents[j] == j:
            depths[j] = 0

    changed = True
    while changed:
        changed = False
        for j in range(n):
            if depths[j] >= 0:
                continue
            p = parents[j]
            if 0 <= p < n and depths[p] >= 0:
                depths[j] = depths[p] + 1
                changed = True

    return depths


# ---------------------------------------------------------------------------
# 3. Graph positional encodings
# ---------------------------------------------------------------------------

def _build_laplacian(parents, norm='none'):
    """Build the graph Laplacian via ``torch_geometric.utils.get_laplacian``.

    Uses PyG's implementation which supports all standard normalization
    variants, matching the pipeline in GRIT / GraphGPS.

    Args:
        parents: Already-parsed (J,) int64 array.
        norm: Laplacian normalization passed to ``get_laplacian``.

            - ``'none'``: combinatorial ``L = D - A``
            - ``'sym'``: symmetric normalized ``L_sym = I - D^{-1/2} A D^{-1/2}``
            - ``'rw'``: random walk normalized ``L_rw = I - D^{-1} A``

    Returns:
        ``np.ndarray`` of shape ``(J, J)`` with dtype float64.
    """
    n = len(parents)
    edge_index = torch.from_numpy(compute_edge_indexs(parents))  # [2, 2*(J-1)]
    pyg_norm = None if norm == 'none' else norm
    lap_index, lap_weight = get_laplacian(edge_index, normalization=pyg_norm,
                                          num_nodes=n)
    L = to_scipy_sparse_matrix(lap_index, lap_weight, num_nodes=n)
    return L.toarray()


def _normalize_eigvecs(eigvecs, eigvals, norm='L2', eps=1e-12):
    """Normalize eigenvectors using various strategies.

    Args:
        eigvecs: ``(J, K)`` eigenvectors.
        eigvals: ``(K,)`` eigenvalues.
        norm: Normalization type — ``'L1'``, ``'L2'``, ``'abs-max'``,
            ``'wavelength'``, ``'sign'`` (sign canonicalization only).

    Returns:
        ``(J, K)`` normalized eigenvectors.
    """
    eigvecs = eigvecs.copy()

    if norm == 'sign':
        for k in range(eigvecs.shape[1]):
            max_idx = np.argmax(np.abs(eigvecs[:, k]))
            if eigvecs[max_idx, k] < 0:
                eigvecs[:, k] *= -1
        return eigvecs

    for k in range(eigvecs.shape[1]):
        col = eigvecs[:, k]
        if norm == 'L1':
            denom = np.abs(col).sum()
        elif norm == 'L2':
            denom = np.sqrt((col ** 2).sum())
        elif norm == 'abs-max':
            denom = np.abs(col).max()
        elif norm == 'wavelength':
            denom = np.abs(col).max()
            lam = np.sqrt(max(eigvals[k], eps))
            denom = denom * lam * 2.0 / np.pi
        else:
            raise ValueError(f"Unsupported eigvec norm: '{norm}'. "
                             f"Choose from 'sign', 'L1', 'L2', 'abs-max', 'wavelength'.")
        if denom > eps:
            eigvecs[:, k] = col / denom

    return eigvecs


def compute_laplacian_eigenvectors(parents, max_freqs=8,
                                   laplacian_norm='sym', eigvec_norm='L2'):
    """Compute the smallest eigenvectors of the graph Laplacian.

    Follows the same decomposition pipeline as GRIT / GraphGPS:
    build the Laplacian, run eigendecomposition, clamp eigenvalues for
    numerical stability, and normalize eigenvectors.

    Args:
        parents: Parent indices (J,).
        max_freqs: Maximum number of frequencies to return. Clamped to
            J-1 if the skeleton has fewer joints.
        laplacian_norm: Laplacian type — ``'none'`` (combinatorial
            ``L = D - A``), ``'sym'`` (symmetric normalized), or
            ``'rw'`` (random walk normalized).
        eigvec_norm: Eigenvector normalization — ``'sign'`` (canonical
            sign only, default), ``'L1'``, ``'L2'``, ``'abs-max'``, or
            ``'wavelength'``.

    Returns:
        Tuple of:

        - ``eigvecs``: ``(J, max_freqs)`` float32 — spectral coordinates.
          When the skeleton has fewer than ``max_freqs + 1`` joints, only
          ``J - 1`` real eigenvectors exist and the trailing columns are
          zero-padded so downstream spectral encoders see a fixed width.
        - ``eigvals``: ``(max_freqs,)`` float32 — corresponding eigenvalues
          (ascending, starting from the Fiedler value), zero-padded to match.
    """
    parents = _parse_parents(parents)
    n = len(parents)
    k = min(n - 1, max_freqs)

    laplacian = _build_laplacian(parents, norm=laplacian_norm.lower())
    eigvals_all, eigvecs_all = np.linalg.eigh(laplacian)

    # Keep the smallest non-trivial eigenvectors (skip trivial eigenvalue ~ 0)
    eigvals = eigvals_all[1:1 + k]
    eigvecs = np.real(eigvecs_all[:, 1:1 + k])

    # Clamp eigenvalues >= 0 for numerical stability
    eigvals = np.maximum(eigvals, 0.0)

    # Normalize eigenvectors
    eigvecs = _normalize_eigvecs(eigvecs, eigvals, norm=eigvec_norm)

    # Right-pad the frequency axis to max_freqs so every skeleton yields the
    # same width regardless of joint count. Padding columns are zero, which
    # is a no-op through WIRE (v=0 → angle=0) and a sample-global constant
    # offset through SignNet (cancels in RoPE attention dot products).
    if k < max_freqs:
        eigvecs = np.pad(eigvecs, ((0, 0), (0, max_freqs - k)))
        eigvals = np.pad(eigvals, (0, max_freqs - k))

    return eigvecs.astype(np.float32), eigvals.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Pairwise relations and topology distances
# ---------------------------------------------------------------------------

EDGE_TYPES = {
    'self': 0, 'parent': 1, 'child': 2, 'sibling': 3,
    'no_relation': 4, 'end_effector': 5, 'ts_token_conn': 6,
}


def compute_edge_relations_and_distances(parents, max_path_len=5):
    """Compute pairwise edge relation types and topology distances.

    Edge relation types:
        0=self, 1=parent, 2=child, 3=sibling, 4=no_relation,
        5=end_effector, 6=ts_token_conn

    Args:
        parents: Parent indices (J,).
        max_path_len: Maximum topology distance (longer paths are clamped).

    Returns:
        Tuple of ``(edge_rel, topo_rel)``, each ``(J, J)`` int16.
    """
    parents = list(parents)
    n = len(parents)

    children = _children_map(parents)

    # --- Edge relations ---
    edge_rel = np.full((n, n), EDGE_TYPES['no_relation'], dtype=np.int16)
    for i in range(n):
        pi = parents[i]
        for j in range(n):
            pj = parents[j]
            if i == j:
                edge_rel[i, j] = EDGE_TYPES['self']
            elif pj == i:
                edge_rel[i, j] = EDGE_TYPES['child']
            elif j == pi and pi != -1:
                edge_rel[i, j] = EDGE_TYPES['parent']
            elif pi != -1 and pj == pi:
                edge_rel[i, j] = EDGE_TYPES['sibling']

        if len(children[i]) == 0:
            edge_rel[i, i] = EDGE_TYPES['end_effector']

    # --- Topology distances via BFS ---
    adj = _adjacency_list(parents)

    topo_rel = np.full((n, n), max_path_len, dtype=np.int16)
    for s in range(n):
        dist = np.full(n, 32767, dtype=np.int16)
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= max_path_len:
                continue
            for v in adj[u]:
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        topo_rel[s] = np.minimum(dist, max_path_len)

    return edge_rel, topo_rel
