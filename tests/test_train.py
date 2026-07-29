import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir

from noda.models.fno import FNO2d
from noda.models.train import build_model, make_train_step, run
from noda.utils.io import save_trajectory
from noda.utils.seed import KeyPurpose, derive_key

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "configs")
GRID = 16


def _tiny_model(key):
    return FNO2d(grid_size=GRID, in_channels=3, width=8, modes1=4, modes2=4, n_layers=2, proj_channels=16, key=key)


def test_train_step_decreases_loss_on_fixed_batch():
    """Directly exercises the optimization mechanics (rollout_loss + train_step) on a
    fixed synthetic batch, independent of data loading/config -- the most direct check
    that gradient descent is actually wired up correctly.
    """
    import optax

    model = _tiny_model(jax.random.PRNGKey(0))
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer)

    key_w0, key_targets = jax.random.split(jax.random.PRNGKey(1))
    w0_batch = jax.random.normal(key_w0, (4, GRID, GRID), dtype=jnp.float32)
    targets_batch = jax.random.normal(key_targets, (4, 3, GRID, GRID), dtype=jnp.float32)

    losses = []
    for _ in range(20):
        model, opt_state, loss = train_step(model, opt_state, w0_batch, targets_batch)
        losses.append(float(loss))

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def _write_fake_split(data_dir, split, n_traj, traj_len, grid):
    split_dir = data_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n_traj):
        traj = rng.normal(size=(traj_len, grid, grid)).astype(np.float32)
        save_trajectory(split_dir / f"traj_{i:04d}.npz", traj, {"split": split, "traj_index": i})


def test_run_end_to_end_smoke(tmp_path):
    """Full run() pipeline on tiny synthetic data: data loading, training loop,
    checkpointing. Confirms the pieces integrate, not just that each works alone.
    """
    data_dir = tmp_path / "data"
    _write_fake_split(data_dir, "train", n_traj=3, traj_len=20, grid=GRID)
    _write_fake_split(data_dir, "val", n_traj=2, traj_len=20, grid=GRID)
    checkpoint_dir = tmp_path / "checkpoints"

    overrides = [
        f"paths.output_dir={data_dir}",
        f"paths.model_dir={checkpoint_dir}",
        f"physics.grid_size={GRID}",
        "model.width=8",
        "model.modes1=4",
        "model.modes2=4",
        "model.n_layers=2",
        "model.proj_channels=16",
        "train.rollout_length=2",
        "train.val_rollout_length=4",
        "train.batch_size=2",
        "train.max_steps=10",
        "train.val_every=5",
        "train.checkpoint_every=10",
        "train.early_stop_patience=1000",
        "train.wandb.enabled=false",
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    model = run(cfg)

    latest_path = checkpoint_dir / f"fno_seed{cfg.seed}_latest.eqx"
    best_path = checkpoint_dir / f"fno_seed{cfg.seed}_best.eqx"
    assert latest_path.exists()
    assert best_path.exists()

    # checkpoint round-trip: reloading must reproduce identical predictions
    template = build_model(cfg, derive_key(cfg.seed, KeyPurpose.MODEL_INIT))
    reloaded = eqx.tree_deserialise_leaves(latest_path, template)
    w = jnp.zeros((GRID, GRID), dtype=jnp.float32)
    assert jnp.array_equal(model(w), reloaded(w))
