import numpy as np

from rl.ppo import ObservationNormalizer


def test_observation_normalizer_tracks_mean_variance_and_clips() -> None:
    normalizer = ObservationNormalizer(
        observation_dim=2,
        clip_value=3.0,
        epsilon=1e-8,
    )
    values = np.array([[1.0, 10.0], [3.0, 14.0]])

    normalizer.update(values)

    np.testing.assert_allclose(normalizer.mean, [2.0, 12.0])
    np.testing.assert_allclose(normalizer.variance, [1.0, 4.0])
    np.testing.assert_allclose(
        normalizer.normalize(values),
        [[-1.0, -1.0], [1.0, 1.0]],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        normalizer.normalize(np.array([22.0, 12.0])),
        [3.0, 0.0],
    )


def test_observation_normalizer_merges_batches() -> None:
    normalizer = ObservationNormalizer(
        observation_dim=1,
        clip_value=10.0,
        epsilon=1e-8,
    )

    normalizer.update(np.array([[0.0], [2.0]]))
    normalizer.update(np.array([[4.0], [6.0]]))

    np.testing.assert_allclose(normalizer.mean, [3.0])
    np.testing.assert_allclose(normalizer.variance, [5.0])
    assert normalizer.count == 4.0
