"""From-scratch MACE interatomic potential and distributed training stack."""

import torch as _torch

# e3nn 0.4.4 (pinned by mace-torch) loads a constants file that contains Python
# `slice` objects; torch >= 2.6 only unpickles allow-listed types by default.
_torch.serialization.add_safe_globals([slice])

__version__ = "0.1.0"
