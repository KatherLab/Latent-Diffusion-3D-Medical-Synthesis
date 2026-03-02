from __future__ import annotations

import torch


def dict_collate(batch):
    """Safe collate for dict batches with tensors + strings."""
    if len(batch) == 0:
        return {}
    if not isinstance(batch[0], dict):
        raise TypeError("Expected batch of dicts")

    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out
