"""Quaternion and rotation utilities."""

from dataclasses import dataclass, field

import numpy as np

from common.dimensions import QDIMS
from common.types import FloatMatrix, FloatVector


@dataclass(frozen=True, init=False)
class Quaternion:
    """Scalar-first quaternion value object."""

    w: float
    x: float
    y: float
    z: float
    _is_normalized: bool = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        w: float,
        x: float,
        y: float,
        z: float,
        *,
        normalize: bool = True,
    ) -> None:
        values = [float(w), float(x), float(y), float(z)]
        if normalize:
            norm = float(np.sqrt(sum(value * value for value in values)))
            if not np.isfinite(norm) or norm == 0.0:
                raise ValueError("quaternion must have a finite, non-zero norm")
            values = [value / norm for value in values]
        object.__setattr__(self, "w", values[0])
        object.__setattr__(self, "x", values[1])
        object.__setattr__(self, "y", values[2])
        object.__setattr__(self, "z", values[3])
        object.__setattr__(self, "_is_normalized", normalize)

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_array(
        cls,
        value: FloatVector,
        *,
        normalize: bool = True,
    ) -> "Quaternion":
        if not isinstance(value, np.ndarray):
            raise TypeError("quaternion must be a NumPy array")
        if value.shape != (QDIMS.quaternion_dim,):
            raise ValueError(
                "quaternion must have shape "
                f"({QDIMS.quaternion_dim},), got {value.shape}"
            )
        w, x, y, z = value
        return cls(w, x, y, z, normalize=normalize)

    def as_array(self) -> FloatVector:
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)

    def norm(self) -> float:
        return float(np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2))

    def to_yaw(self) -> float:
        return float(
            np.arctan2(
                2.0 * (self.w * self.z + self.x * self.y),
                1.0 - 2.0 * (self.y * self.y + self.z * self.z),
            )
        )

    def normalized(self) -> "Quaternion":
        return Quaternion(self.w, self.x, self.y, self.z)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        if not isinstance(other, Quaternion):
            raise TypeError("quaternion can only multiply another Quaternion")
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            normalize=self._is_normalized and other._is_normalized,
        )

    def to_rotation_matrix(self) -> FloatMatrix:
        if not np.isclose(self.norm(), 1.0):
            raise ValueError("quaternion must have unit norm")

        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
                 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
                 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
                 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
