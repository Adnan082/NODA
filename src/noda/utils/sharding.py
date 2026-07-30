"""Single-host, multi-device data parallelism via JAX's sharding API (not `pmap`,
which is the older, superseded mechanism). Degrades to a no-op on one device, so the
exact same code path runs unchanged on this laptop's single CPU device and on a real
multi-GPU box -- no separate single/multi-device branch anywhere else in the codebase.

How it works: batch-dimension arrays get sharded (split) across devices via
`data_sharding`; the model and optimizer state get replicated (an identical full copy
on every device) via `replicated_sharding`. Once inputs carry that sharding, XLA's
GSPMD partitioner automatically parallelizes any `jax.jit`-compiled computation over
them -- including gradient averaging, which becomes an implicit all-reduce across
devices -- without `train_step` itself needing to know how many devices exist.
"""
from __future__ import annotations

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def make_data_parallel_shardings() -> tuple[Mesh, NamedSharding, NamedSharding]:
    """Returns (mesh, data_sharding, replicated_sharding) built from every locally
    visible device (`jax.devices()`) -- 1 on this laptop's CPU, 4 on the AWS box.
    """
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("batch",))
    data_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    return mesh, data_sharding, replicated_sharding
