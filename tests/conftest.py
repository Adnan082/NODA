import os

# Force CPU: deterministic, reproducible test results. GPU float reductions are
# not guaranteed bit-associative, which would break the bit-identical
# reproducibility test.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
