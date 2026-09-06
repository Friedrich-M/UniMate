"""Geodesic distance between rotation matrices."""

import torch as th


def geodesic_distance(R_pred, R_gt):
    """Compute geodesic distance (angle in radians) between rotation matrices.

    Args:
        R_pred: Predicted rotation matrices (..., 3, 3).
        R_gt: Ground truth rotation matrices (..., 3, 3).

    Returns:
        Geodesic angle in radians (..., 1).
    """
    R_rel = th.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = th.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)

    epsilon = 1e-6
    theta = th.arccos(th.clamp((trace - 1) / 2, -1 + epsilon, 1 - epsilon))

    return theta[..., None]
