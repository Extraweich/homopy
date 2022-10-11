import matplotlib.pyplot as plt
from Stiffness import *
from Methods import *
from fiberpy import *
from fiberoripy import *


if __name__ == "__main__":
    #%%

    Carbon_fiber = Isotropy(242e9, 0.1)
    Glass_fiber = Isotropy(80e9, 0.22)
    Polyamid6 = Isotropy(1.18e9, 0.35)

    MT = Mori_Tanaka(Polyamid6, Carbon_fiber, 0.25, 347)
    MT2 = Mori_Tanaka(Polyamid6, Glass_fiber, 0.25, 225)
    C_eff = MT.get_effective_stiffness()
    C2_eff = MT2.get_effective_stiffness()
    S_eff = np.linalg.inv(C_eff)
    S2_eff = np.linalg.inv(C2_eff)

    from Stiffness_Plot import plot_E_body, polar_plot_E_body, polar_plot
    from Tsai_Hill import *

    TH_Carb = Tsai_Hill(
        242 * 1e9,
        1.18 * 1e9,
        105 * 1e9,
        0.4 * 1e9,
        0.1,
        0.35,
        2.5 * 1e-3,
        7.2 / 2 * 1e-6,
        0.25,
    )
    angles = np.arange(0, 2 * np.pi, 0.01)
    Es_Carb = TH_Carb.get_E(angles)

    # plot_E_body(S_eff, 200, 100,[6e10,6e10,6e10])
    pC = polar_plot_E_body(S_eff, 400, 0, plot=False)
    pG = polar_plot_E_body(S2_eff, 400, 0, plot=False)

    polar_plot(
        [pC + ("MT Carbon",), pG + ("MT Glass",)]
    )  # (angles, Es_Carb, 'Shear-lag')

    from fiberpy.mechanics import *

    rve_data = {
        "rho0": 1.14e-9,
        "E0": 1.18e9,
        "nu0": 0.35,
        "rho1": 2.55e-9,
        "E1": 242e9,
        "nu1": 0.1,
        "vf": 0.25,
        "aspect_ratio": 347,
    }
    fiber = FiberComposite(rve_data)

    Esh_own = MT.eshelby66
    Esh_pack = MT.tensor2mandel(fiber.Eshelby())

    #%% Orientierung
    from fiberoripy.closures import (
        IBOF_closure,
        compute_closure,
        hybrid_closure,
        linear_closure,
        quadratic_closure,
    )

    N2 = np.eye(3)
    N2[0, 0] = 19 / 32
    N2[1, 1] = 10 / 32
    N2[2, 2] = 3 / 32
    # N4 = quadratic_closure(N2)
    N4 = IBOF_closure(N2)

    ud = fiber.MoriTanaka()
    ud_ave = fiber.ABar(
        np.array([0.5, 0.5, 0.0]), model="MoriTanaka", closure="invariants"
    )
    ud_inv = np.linalg.inv(ud)
    # plot_E_body(ud_inv, 200, 100,[6e10,6e10,6e10])

    C_eff_ave = MT.get_average_stiffness(C_eff, N2, N4)
    S_eff_ave = np.linalg.inv(C_eff_ave)
    S_mat = np.linalg.inv(Polyamid6.stiffness66)
    p2 = polar_plot_E_body(S_eff_ave, 400, 0, plot=False)
    p3 = polar_plot_E_body(S_mat, 400, 0, plot=False)
    polar_plot([pC + ("MT UD",), p2 + ("MT planar iso",), p3 + ("PA6",)])

    #%% Hybrid
    MTH = Mori_Tanaka(
        Polyamid6, [Glass_fiber, Carbon_fiber], [0.125, 0.125], [225, 347]
    )
    C_eff_H = MTH.get_effective_stiffness()
    S_eff_H = np.linalg.inv(C_eff_H)

    pH = polar_plot_E_body(S_eff_H, 400, 0, plot=False)
    polar_plot([pG + ("MT Glass",), pC + ("MT Carbon",), pH + ("MT Hybrid",)])
