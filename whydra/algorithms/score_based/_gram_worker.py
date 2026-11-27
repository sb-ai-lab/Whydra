"""Read-only numerical GES batches for the shared process executor."""


def score_batch(data, gram, safe_columns, design_conditions, keys):
    from .scores import LocalScoreCache

    # No per-fit arrays or score tables survive this call in a worker. Shared
    # memory leases and persistent worker ownership belong to process_map.
    cache = LocalScoreCache._from_gaussian_statistics(
        data, gram, safe_columns, design_conditions
    )
    return cache._score_batch(keys)
