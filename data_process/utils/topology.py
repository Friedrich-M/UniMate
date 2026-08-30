"""Skeleton topology utilities.

All public functions accept a parent index array ``(J,)`` (list or numpy
array) where ``parents[j]`` is the parent of joint *j* and the root has
parent ``-1``.

Sections:
    1. Graph representations   — adjacency, edge index, joint depths
    2. Spectral encodings      — Laplacian eigenvectors
    3. Pairwise relations      — edge types, topology distances
    4. Kinematic chains
"""

import bisect
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_parents(parents):
    """Normalize *parents* to a 1-D int64 numpy array."""
    return np.asarray(parents, dtype=np.int64).ravel()


def _children_map(parents):
    """List-of-lists mapping each joint to its children."""
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
    """Undirected adjacency list built from *parents*."""
    n = len(parents)
    adj = [[] for _ in range(n)]
    for c, p in enumerate(parents):
        if p is None or p == -1:
            continue
        adj[c].append(p)
        adj[p].append(c)
    return adj


# ---------------------------------------------------------------------------
# 1. Graph representations
# ---------------------------------------------------------------------------

def compute_adjacency_matrix(parents, self_loops=True):
    """Dense undirected adjacency matrix ``(J, J)`` (float32)."""
    parents = _parse_parents(parents)
    n = len(parents)
    adj = np.eye(n, dtype=np.float32) if self_loops else np.zeros((n, n), dtype=np.float32)
    for child, parent in enumerate(parents):
        if child == parent or parent < 0:
            continue
        adj[child, parent] = 1.0
        adj[parent, child] = 1.0
    return adj


def compute_edge_indexs(parents):
    """PyG-style edge index ``[2, 2*(J-1)]`` (int64).

    Undirected: includes both parent->child and child->parent edges.
    Self-loops are excluded.
    """
    parents = _parse_parents(parents)
    children = np.arange(len(parents))
    valid = (children != parents) & (parents >= 0)
    p = parents[valid]
    c = children[valid]
    return np.stack([np.concatenate([p, c]), np.concatenate([c, p])], axis=0)


def compute_joint_depths(parents):
    """Depth of each joint in the kinematic tree ``(J,)`` (int64, root=0).

    Handles arbitrary joint ordering (parent index may be > child index).
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
# 2. Spectral encodings
# ---------------------------------------------------------------------------

def _build_laplacian(parents, norm='none'):
    """Graph Laplacian ``(J, J)`` of the undirected skeleton graph.

    Args:
        parents: Already-parsed (J,) int64 array.
        norm: ``'none'`` (combinatorial ``L = D - A``), ``'sym'``
            (``I - D^{-1/2} A D^{-1/2}``) or ``'rw'`` (``I - D^{-1} A``).
    """
    adj = compute_adjacency_matrix(parents, self_loops=False).astype(np.float64)
    deg = adj.sum(axis=1)
    if norm == 'none':
        return np.diag(deg) - adj
    with np.errstate(divide='ignore'):
        if norm == 'sym':
            d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
            return np.eye(len(deg)) - d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
        if norm == 'rw':
            d_inv = np.where(deg > 0, 1.0 / deg, 0.0)
            return np.eye(len(deg)) - d_inv[:, None] * adj
    raise ValueError(f"Unsupported laplacian norm: '{norm}'. Choose from 'none', 'sym', 'rw'.")


def _normalize_eigvecs(eigvecs, eigvals, norm='L2', eps=1e-12):
    """Normalize eigenvector columns (``'sign'``, ``'L1'``, ``'L2'``,
    ``'abs-max'`` or ``'wavelength'``)."""
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
    """Smallest non-trivial eigenvectors of the graph Laplacian.

    Follows the GRIT / GraphGPS decomposition: build the Laplacian, run an
    eigendecomposition, clamp eigenvalues and normalize eigenvectors.

    Args:
        parents: Parent indices (J,).
        max_freqs: Number of frequencies to return (clamped to J-1).
        laplacian_norm: ``'none'``, ``'sym'`` or ``'rw'``.
        eigvec_norm: ``'sign'``, ``'L1'``, ``'L2'``, ``'abs-max'`` or
            ``'wavelength'``.

    Returns:
        ``(eigvecs (J, max_freqs) float32, eigvals (max_freqs,) float32)``.
        Skeletons with fewer than ``max_freqs + 1`` joints are zero-padded on
        the frequency axis so every skeleton yields the same width.
    """
    parents = _parse_parents(parents)
    n = len(parents)
    k = min(n - 1, max_freqs)

    laplacian = _build_laplacian(parents, norm=laplacian_norm.lower())
    eigvals_all, eigvecs_all = np.linalg.eigh(laplacian)

    # Skip the trivial (constant) eigenvector at eigenvalue ~0.
    eigvals = np.maximum(eigvals_all[1:1 + k], 0.0)
    eigvecs = np.real(eigvecs_all[:, 1:1 + k])
    eigvecs = _normalize_eigvecs(eigvecs, eigvals, norm=eigvec_norm)

    if k < max_freqs:
        eigvecs = np.pad(eigvecs, ((0, 0), (0, max_freqs - k)))
        eigvals = np.pad(eigvals, (0, max_freqs - k))

    return eigvecs.astype(np.float32), eigvals.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Pairwise relations and topology distances
# ---------------------------------------------------------------------------

EDGE_TYPES = {
    'self': 0, 'parent': 1, 'child': 2, 'sibling': 3,
    'no_relation': 4, 'end_effector': 5, 'ts_token_conn': 6,
}


def compute_edge_relations_and_distances(parents, max_path_len=5):
    """Pairwise edge relation types and clamped topology distances.

    Edge relation types follow ``EDGE_TYPES``: 0=self, 1=parent, 2=child,
    3=sibling, 4=no_relation, 5=end_effector, 6=ts_token_conn.

    Returns:
        ``(edge_rel, topo_rel)``, each ``(J, J)`` int16.
    """
    parents = list(parents)
    n = len(parents)
    children = _children_map(parents)

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


# ---------------------------------------------------------------------------
# 4. Kinematic chain decomposition
# ---------------------------------------------------------------------------

def compute_kinematic_chains(parents, policy='h_first', root=0):
    """Decompose a skeleton tree into kinematic chains (root-to-leaf paths).

    Args:
        parents: Parent indices (J,).
        policy: Branching priority — ``'h_first'`` (highest index first)
            or ``'l_first'`` (lowest index first).
        root: Index of the root joint.

    Returns:
        List of chains, each a list of joint indices from root to leaf.
    """
    parents = list(parents)
    n = len(parents)
    children_dict = {i: [] for i in range(n)}

    for j in range(n):
        p = parents[j]
        if j == root or p is None or p == -1:
            continue
        if not (0 <= p < n):
            raise ValueError(f"Invalid parent index parents[{j}]={p}")
        if policy == 'h_first':
            _reverse_insort(children_dict[p], j)
        else:
            bisect.insort(children_dict[p], j)

    chains = []
    _build_kinchains([], root, children_dict, chains, policy)
    return chains


def _reverse_insort(a, x, lo=0, hi=None):
    """Insert *x* into descending-sorted list *a*, keeping order."""
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x > a[mid]:
            hi = mid
        else:
            lo = mid + 1
    a.insert(lo, x)


def _build_kinchains(chain, j, children_dict, chains, policy):
    """Recursively build kinematic chains from the skeleton tree."""
    children = children_dict[j]
    chain2 = chain + [j]

    if len(children) == 0:
        chains.append(chain2)
        return
    if len(children) == 1:
        _build_kinchains(chain2, children[0], children_dict, chains, policy)
        return

    main_child = children[0]
    for child in children:
        if child == main_child:
            _build_kinchains(chain2, child, children_dict, chains, policy)
        else:
            _build_kinchains([j], child, children_dict, chains, policy)
