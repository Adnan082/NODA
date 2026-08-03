import os

# Force CPU: deterministic, reproducible test results. GPU float reductions are
# not guaranteed bit-associative, which would break the bit-identical
# reproducibility test.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# Force a non-interactive matplotlib backend: tests only ever savefig() to disk, but
# without this matplotlib may pick an interactive backend (e.g. TkAgg) that tries to
# open a real GUI window on first plt.subplots() call -- fails outright on a machine
# with no/broken Tk install, and is pointless overhead even where Tk does work.
import matplotlib

matplotlib.use("Agg")
