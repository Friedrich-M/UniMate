"""6D rotation representation → rotation matrix conversions.

The 6D representation is from Zhou et al., "On the Continuity of Rotation
Representations in Neural Networks", CVPR 2019
(http://arxiv.org/abs/1812.07035): the first two rows of the rotation matrix,
orthonormalized via Gram-Schmidt on decode.

Convention: rotation matrices act on column vectors via post-multiplication,
i.e. ``transformed_point = R @ point``.
"""

import numpy as np
import torch


def rotation_6d_to_matrix_safe(cont6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to matrix with NaN protection and epsilon offset.

    Args:
        cont6d: 6D rotation representation (*, 6).

    Returns:
        Rotation matrices (*, 3, 3).
    """
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    epsilon = 1e-8
    cont6d = torch.nan_to_num(cont6d) + epsilon
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / torch.linalg.norm(x_raw, dim=-1, keepdims=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.linalg.norm(z, dim=-1, keepdims=True)
    y = torch.cross(z, x, dim=-1)
    return torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)


def rotation_6d_to_matrix_np(cont6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation to matrix (numpy version).

    Args:
        cont6d: 6D rotation representation (*, 6).

    Returns:
        Rotation matrices (*, 3, 3).
    """
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw, axis=-1)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
