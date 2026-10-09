import numpy as np

from ...background_knowledge import MatrixEncoding, endpoint_codes


def to_circle_adjacency(mat: np.ndarray, encoding: MatrixEncoding | str) -> np.ndarray:
    """
    Return a copy where every existing edge is represented as o-o, and absent
    edges remain 0.

    Кодировка называется явно: сейчас вся библиотека работает в кодировке
    causal-learn, но параметр сохранён, потому что BackgroundKnowledge и
    conversions умеют принимать и матрицы в кодировке pcalg (например, из R).
    """
    _null, _tail, _arrow, circle = endpoint_codes(encoding)
    out = np.array(mat, dtype=int, copy=True)
    out[out != 0] = circle
    return out
