from typing import List

import numpy as np


def copy_batch_block_offsets(
    dst_tensor: np.ndarray,
    batch_size: int,
    batch_cache_indices: List[np.ndarray],
    num_pools: int,
    offset: int,
):
    for pool_idx in range(num_pools):
        for idx in range(batch_size):
            batch_idx = pool_idx * batch_size + idx
            batch_cache_index = batch_cache_indices[batch_idx]
            dst_tensor[pool_idx, idx + offset, 0, : len(batch_cache_index)] = batch_cache_index
            dst_tensor[pool_idx, idx + offset, 1, : len(batch_cache_index)] = batch_cache_index + 1
