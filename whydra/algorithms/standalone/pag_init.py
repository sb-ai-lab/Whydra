import numpy as np

from ...background_knowledge import MatrixEncoding, endpoint_codes


def to_circle_adjacency(mat: np.ndarray, encoding: MatrixEncoding | str) -> np.ndarray:
    """Copy the matrix with circle marks at both ends of each edge and zeros for absent edges.

    Whydra uses causal-learn encoding internally. The encoding parameter is explicit
    because BackgroundKnowledge and the conversion functions also accept pcalg
    matrices, for example from R.
    """
    _null, _tail, _arrow, circle = endpoint_codes(encoding)
    out = np.array(mat, dtype=int, copy=True)
    out[out != 0] = circle
    return out
