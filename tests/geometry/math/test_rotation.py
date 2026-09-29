import numpy as np
import pytest

from geometry.math.rotation import Quaternion


def test_quaternion_normalizes_by_default() -> None:
    result = Quaternion(2.0, 0.0, 0.0, 0.0)

    np.testing.assert_allclose(result.as_array(), [1.0, 0.0, 0.0, 0.0])
    assert result.norm() == 1.0


def test_unnormalized_quaternion_requires_explicit_option() -> None:
    result = Quaternion(2.0, 0.0, 0.0, 0.0, normalize=False)

    assert result.norm() == 2.0
    np.testing.assert_allclose(result.normalized().as_array(), [1.0, 0.0, 0.0, 0.0])


def test_normalized_quaternion_rejects_zero_norm() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        Quaternion(0.0, 0.0, 0.0, 0.0)


def test_quaternion_from_array_requires_numpy_array_and_shape() -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        Quaternion.from_array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        Quaternion.from_array(np.ones(3))


def test_quaternion_multiplication_identity_both_orders() -> None:
    identity = Quaternion.identity()
    quaternion = Quaternion(0.5, 0.5, 0.5, 0.5)

    np.testing.assert_allclose(
        (identity * quaternion).as_array(),
        quaternion.as_array(),
    )
    np.testing.assert_allclose(
        (quaternion * identity).as_array(),
        quaternion.as_array(),
    )


def test_raw_quaternion_product_stays_raw() -> None:
    result = Quaternion.identity() * Quaternion(0.0, 2.0, 0.0, 0.0, normalize=False)

    np.testing.assert_allclose(result.as_array(), [0.0, 2.0, 0.0, 0.0])


def test_identity_quaternion_gives_identity_matrix() -> None:
    np.testing.assert_allclose(Quaternion.identity().to_rotation_matrix(), np.eye(3))


def test_ninety_degree_positive_yaw_rotates_x_to_y() -> None:
    half_angle = np.sqrt(0.5)
    quaternion = Quaternion(half_angle, 0.0, 0.0, half_angle)
    rotation = quaternion.to_rotation_matrix()

    np.testing.assert_allclose(
        rotation @ [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_to_yaw_returns_positive_ninety_degree_heading() -> None:
    half_angle = np.sqrt(0.5)

    assert np.isclose(
        Quaternion(half_angle, 0.0, 0.0, half_angle).to_yaw(),
        np.pi / 2.0,
    )


def test_rotation_matrix_is_orthonormal_with_positive_determinant() -> None:
    rotation = Quaternion(0.5, 0.5, 0.5, 0.5).to_rotation_matrix()

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_rotation_matrix_rejects_raw_non_unit_quaternion() -> None:
    quaternion = Quaternion(2.0, 0.0, 0.0, 0.0, normalize=False)

    with pytest.raises(ValueError, match="unit norm"):
        quaternion.to_rotation_matrix()
