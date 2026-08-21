"""Adsorbate placement helpers."""

from __future__ import annotations

import numpy as np


def place_adsorbate(cluster, adsorbate, indices, height=None, orientation=None):
    """Place an adsorbate on a cluster using one to four binding indices."""
    cluster_center = cluster.get_center_of_mass()
    indices = list(indices)

    if len(indices) == 1:
        pos = cluster.positions[indices[0]]
        normal = pos - cluster_center
    elif len(indices) == 2:
        p1, p2 = cluster.positions[indices]
        pos = (p1 + p2) / 2.0
        edge_vec = p2 - p1
        center_vec = pos - cluster_center
        normal = np.cross(edge_vec, np.cross(center_vec, edge_vec))
        if np.linalg.norm(normal) < 1e-6:
            normal = center_vec.copy()
        if np.dot(normal, center_vec) < 0:
            normal = -normal
    elif len(indices) == 3:
        p1, p2, p3 = cluster.positions[indices]
        pos = (p1 + p2 + p3) / 3.0
        normal = np.cross(p2 - p1, p3 - p1)
        if np.dot(normal, pos - cluster_center) < 0:
            normal = -normal
    elif len(indices) == 4:
        p1, p2, p3, p4 = cluster.positions[indices]
        pos = (p1 + p2 + p3 + p4) / 4.0
        normal = np.cross(p2 - p1, p3 - p1)
        if np.dot(normal, pos - cluster_center) < 0:
            normal = -normal
    else:
        raise ValueError("indices should contain 1-4 elements")

    if orientation is not None:
        orientation = np.asarray(orientation, float)
        if np.linalg.norm(orientation) < 1e-6:
            raise ValueError("orientation vector has zero length")
        normal = orientation

    normal = normal / np.linalg.norm(normal)
    if height is None:
        height = 2.0

    ads = adsorbate.copy()
    if len(ads) == 0:
        raise ValueError("adsorbate is empty")

    anchor_pos = ads.positions[0].copy()
    ads.positions -= anchor_pos
    if len(ads) > 1:
        ads.rotate(ads.positions[1], normal, center=(0, 0, 0))
    ads.translate(pos + normal * height)

    combined = cluster.copy()
    combined += ads
    return combined
