import jax.numpy as jnp
import numpy as np

from noda.eval.metrics import crps_fair, rank_histogram, spread_skill_ratio


def test_spread_skill_ratio_hand_computed():
    """N=4 members [1,2,3,4] (sample variance=5/3), truth=3.0.
    mean_var = 5/3, mse_of_mean = (2.5-3.0)^2 = 0.25.
    ratio = sqrt((5/4)*(5/3)) / sqrt(0.25) = sqrt(25/12) / 0.5 ~= 1.44338 / 0.5 ~= 2.88675.
    """
    ensembles = jnp.array([[1.0, 2.0, 3.0, 4.0]]).reshape(1, 4)  # (T=1, N=4)
    truths = jnp.array([3.0]).reshape(1)  # (T=1,)
    ratio = spread_skill_ratio(ensembles, truths)
    assert np.isclose(ratio, 2.88675, atol=1e-4)


def test_spread_skill_ratio_below_one_means_overconfident():
    """A tight ensemble (small spread) that's still far from the truth should give
    ratio << 1 -- the defining property of the metric, not just a formula check.
    """
    ensembles = jnp.array([[1.0, 1.01, 0.99, 1.0]]).reshape(1, 4)  # nearly zero spread
    truths = jnp.array([5.0]).reshape(1)  # far from the (confident) ensemble
    ratio = spread_skill_ratio(ensembles, truths)
    assert ratio < 0.1


def test_crps_fair_hand_computed():
    """ensemble=[1,2,3], truth=5.
    term1 = mean(|1-5|,|2-5|,|3-5|) = mean(4,3,2) = 3.
    term2 = sum_ij|xi-xj| / (2*3*2) = 8/12 = 2/3 (independent of truth).
    CRPS = 3 - 2/3 = 7/3 ~= 2.33333.
    """
    ensemble = jnp.array([1.0, 2.0, 3.0])
    truth = jnp.array(5.0)
    crps = crps_fair(ensemble, truth)
    assert np.isclose(crps, 7.0 / 3.0, atol=1e-5)


def test_crps_fair_uses_N_times_N_minus_1_not_N_squared():
    """The CLAUDE.md pitfall this formula exists to avoid: with the biased N^2
    denominator, term2 would be 8/18=0.4444 instead of 8/12=0.6667, giving
    CRPS=3-0.4444=2.5556 instead of the correct 2.3333 -- a real, checkable
    difference, not a rounding-level one.
    """
    ensemble = jnp.array([1.0, 2.0, 3.0])
    truth = jnp.array(5.0)
    crps = crps_fair(ensemble, truth)
    biased_wrong_value = 3.0 - 8.0 / 18.0
    assert not np.isclose(crps, biased_wrong_value, atol=1e-3)


def test_rank_histogram_hand_computed():
    ensemble = jnp.array([1.0, 2.0, 3.0])

    # truth below every member -> rank 0
    hist = rank_histogram(ensemble, jnp.array(0.0))
    assert list(hist) == [1, 0, 0, 0]

    # truth between members 2 and 3 (2 members below it) -> rank 2
    hist = rank_histogram(ensemble, jnp.array(2.5))
    assert list(hist) == [0, 0, 1, 0]

    # truth above every member -> rank N=3
    hist = rank_histogram(ensemble, jnp.array(10.0))
    assert list(hist) == [0, 0, 0, 1]


def test_rank_histogram_accumulates_over_multiple_grid_cells():
    """A (N, 2) ensemble (2 'grid cells') with a (2,) truth should produce ranks for
    both cells in one histogram, not just the first.
    """
    ensemble = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # (N=3, 2 cells)
    truth = jnp.array([0.0, 10.0])  # cell 0: below all (rank 0); cell 1: above all (rank 3)
    hist = rank_histogram(ensemble, truth)
    assert list(hist) == [1, 0, 0, 1]
