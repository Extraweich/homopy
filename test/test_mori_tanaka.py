import numpy as np
import pytest
import mechkit
import mechmean
from mechkit.operators import Sym_Fourth_Order_Special
from homopy import methods, elasticity


@pytest.fixture()
def random_complete_sym():
    return Sym_Fourth_Order_Special(label="complete")(np.random.rand(3, 3, 3, 3))


@pytest.mark.parametrize("inclusion_shape", ["ellipsoid", "needle", "sphere"])
def test_mt(random_complete_sym, inclusion_shape, manual_debug=False):

    v_frac = np.random.rand() * 0.74 + 0.01
    a_ratio = np.random.rand() * 200 + 20

    E_fiber = np.random.rand() * 1e10 + 1e9
    E_matrix = np.random.rand() * 2e11 + 1e10

    nu_fiber = np.random.rand() * 0.3 + 0.1
    nu_matrix = np.random.rand() * 0.3 + 0.1

    N4 = random_complete_sym

    # mechmean

    inp = {
        "E_f": E_fiber,
        "E_m": E_matrix,
        "N4": N4,
        "c_f": v_frac,
        "nu_f": nu_fiber,
        "nu_m": nu_matrix,
    }

    fiber_mechmean = mechkit.material.Isotropic(
        E=inp["E_f"],
        nu=inp["nu_f"],
    )

    matrix_mechmean = mechkit.material.Isotropic(
        E=inp["E_m"],
        nu=inp["nu_m"],
    )

    averager = mechmean.orientation_averager.AdvaniTucker(N4=inp["N4"])

    if inclusion_shape == "ellipsoid":
        P_func = mechmean.hill_polarization.Castaneda().spheroid
        hill_polarization = P_func(aspect_ratio=a_ratio, matrix=matrix_mechmean)
    elif inclusion_shape == "needle":
        P_func = mechmean.hill_polarization.Castaneda().needle
        hill_polarization = P_func(matrix=matrix_mechmean)
    elif inclusion_shape == "sphere":
        P_func = mechmean.hill_polarization.Castaneda().sphere
        hill_polarization = P_func(matrix=matrix_mechmean)

    input_dict = {
        "phases": {
            "inclusion": {
                "material": fiber_mechmean,
                "volume_fraction": inp["c_f"],
                "hill_polarization": hill_polarization,
            },
            "matrix": {
                "material": matrix_mechmean,
            },
        },
        "averaging_func": averager.average,
    }

    mt_mechmean = mechmean.approximation.MoriTanakaOrientationAveragedBenveniste(
        **input_dict
    )

    C_eff_mechmean = mt_mechmean.calc_C_eff()

    # HomoPy

    fiber = elasticity.Isotropy(E_fiber, nu_fiber)
    matrix = elasticity.Isotropy(E_matrix, nu_matrix)

    if inclusion_shape == "needle":
        # Aspect ratio for needle shaped inclusion in HomoPy relates to the two minor axis,
        # therefore set to 1 to have a circular cross section
        a_ratio = 1

    mt_homopy = methods.MoriTanaka(
        matrix, fiber, v_frac, a_ratio, N4=N4, shape=inclusion_shape, symmetrize=False
    )

    C_eff_homopy = mt_homopy.effective_stiffness66

    if True:
        # This considers only coefficients in the upper left quadrant and the diagonal of the lower right quadrant
        interesting_indices = np.s_[
            [0, 1, 2, 0, 0, 1, 3, 4, 5], [0, 1, 2, 1, 2, 2, 3, 4, 5]
        ]
        C_eff_homopy = C_eff_homopy[interesting_indices]
        C_eff_mechmean = C_eff_mechmean[interesting_indices]

    coeffcient_difference_maximum = np.max(C_eff_homopy - C_eff_mechmean)
    coefficient_maximum = np.max(C_eff_homopy)
    coefficient_minimum = np.min(C_eff_homopy)

    if not manual_debug:
        print(C_eff_mechmean)
        print(C_eff_homopy)
        print(
            np.linalg.norm(C_eff_homopy - C_eff_mechmean) / np.linalg.norm(C_eff_homopy)
        )

        print(
            f"maximum deviation in one tensor coefficient= {coeffcient_difference_maximum}"
        )
        print(f"maximum tensor coefficient= {coefficient_maximum}")
        print(f"minimum tensor coefficient= {coefficient_minimum}")

        assert np.allclose(
            C_eff_homopy,
            C_eff_mechmean,
            rtol=1e-7,
            atol=1e-7,
        )
    else:

        return coeffcient_difference_maximum


if __name__ == "__main__":
    # For detailed debugging, run this script, e.g. from ipython by "%run test/test_mori_tanaka.py"

    # Make it deterministic
    np.random.seed(16)

    maxima = []
    for i in range(1000):
        N4 = Sym_Fourth_Order_Special(label="complete")(np.random.rand(3, 3, 3, 3))
        maximum = test_mt(N4, inclusion_shape="sphere", manual_debug=True)
        maxima.append(maximum)

    print(f"max(maxima) = {max(maxima)}")
    print(f"mean(maxima) = {np.mean(maxima)}")
