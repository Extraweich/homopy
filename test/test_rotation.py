import numpy as np
import pytest
from homopy import methods


@pytest.fixture()
def random_planar_stiffness():
    mat = np.random.rand(3, 3)
    mat_sym = 1/2*(mat + mat.T)
    return mat_sym


def test_rot(random_planar_stiffness):
    rot_angle = 2*np.pi*np.random.rand()
    rotated_stiffness = methods.Laminate.rotate_stiffness(
        random_planar_stiffness, rot_angle)
    back_rotated_stiffness = methods.Laminate.rotate_stiffness(
        rotated_stiffness, -rot_angle)

    assert np.allclose(
        random_planar_stiffness,
        back_rotated_stiffness,
        rtol=1e-6
    )
