"""Ordered sparse snapshots; endpoint conditions remain live in the rules."""

import numpy as np


def marked_pairs(mask):
    columns, rows = np.nonzero(mask.T)
    return list(zip(rows.tolist(), columns.tolist(), strict=False))


def adjacent_nodes(pag):
    mask = (pag != 0) | (pag.T != 0)
    return [np.flatnonzero(row).tolist() for row in mask]
