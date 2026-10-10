import math
import pathlib

import equinox as eqx
from hydra import compose, initialize_config_dir

from noda.eval.benchmark import run_experiment2
from noda.models.train import build_model
from noda.utils.seed import KeyPurpose, derive_key

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "configs")


def test_run_experiment2_smoke(tmp_path):
    """Confirms the benchmark runs end-to-end (both forward models, cost sweep +
    accuracy comparison) and returns sane numbers, on tiny settings for speed --
    this is a reporting script over already-unit-tested primitives (Day 3's
    enkf_step/run_da), not new core math, so a lighter smoke test is appropriate
    here versus Day 3's exact known-answer tests.
    """
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model.width=4", "model.modes1=4", "model.modes2=4",
                "model.n_layers=1", "model.proj_channels=8",
            ],
        )

    # Exercise real serialisation/loading without requiring an untracked trained
    # checkpoint. Random weights are sufficient for this plumbing smoke test;
    # its outputs are not a model-quality benchmark.
    cfg.paths.model_dir = str(tmp_path)
    model = build_model(cfg, derive_key(cfg.seed, KeyPurpose.MODEL_INIT))
    eqx.tree_serialise_leaves(tmp_path / f"fno_seed{cfg.seed}_best.eqx", model)

    results = run_experiment2(
        cfg, n_ens_sweep=[2, 4], accuracy_n_cycles=2, accuracy_n_ens=4, out_dir=tmp_path
    )

    assert len(results["cost"]) == 4  # 2 forward models x 2 n_ens values
    for r in results["cost"]:
        assert r["seconds_per_cycle"] > 0
        assert r["usd_per_cycle"] > 0

    assert set(results["accuracy"].keys()) == {"numerical", "surrogate"}
    for rmse in results["accuracy"].values():
        assert math.isfinite(rmse) and rmse > 0

    assert (tmp_path / "experiment2_table.txt").exists()
    assert (tmp_path / "experiment2_cost.png").exists()

