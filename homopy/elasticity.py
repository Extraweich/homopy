# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:09:24 2022

@author: nicolas.christ@kit.edu

Module that contains the linear elastic stiffness classes of Isotropy and Transverse Isotropy.
"""

import numpy as np
from .tensor import Tensor


class Elasticity(Tensor):
    """
    Elasticity class to express generic elasitc stiffness tensors. The class
    inherits from the Tensor class.

    Attributes
    ----------
    stiffness3333 : ndarray of shape (3, 3, 3, 3)
        Stiffness values in the regular tensor notation in Pa.
    stiffness66 : ndarray of shape (6, 6)
        Stiffness values in the normalized Voigt notation in Pa.

    """

    def __init__(self):
        super().__init__()
        self.stiffness3333 = np.zeros((3, 3, 3, 3))
        self.stiffness66 = np.zeros((6, 6))


class TransverseIsotropy(Elasticity):
    """
    Transverse Isotropy class to express transverse-isotropic elasitc stiffness tensors.
    The class inherits from the Elasticity class.

    Parameters
    ----------
    E1 : float
        Young's modulus in longitudinal direction in Pa.
    E2 : float
        Young's modulus in transverse direction in Pa.
    G12 : float
        Shear modulus in the longitudinal-transverse plane in Pa.
    G23 : float
        Shear modulus in the transverse-transverse plane in Pa.
    nu12 : float
        Poisson's ratio in longitudinal direction (dimensionless).

    Attributes
    ----------
    E1 : float
        Young's modulus in longitudinal direction in Pa.
    E2 : float
        Young's modulus in transverse direction in Pa.
    G12 : float
        Shear modulus in the longitudinal-transverse plane in Pa.
    G23 : float
        Shear modulus in the transverse-transverse plane in Pa.
    nu12 : float
        Poisson's ratio in longitudinal direction (dimensionless).
    nu23 : float
        Poisson's ratio in transverse direction (dimensionless).

    """

    def __init__(self, E1, E2, G12, G23, nu12):
        super().__init__()
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.G23 = G23
        self.nu12 = nu12
        self.nu21 = self.E2 / self.E1 * self.nu12
        self.nu23 = self.E2 / (2 * self.G23) - 1
        self._get_stiffness()

    def _get_stiffness(self):
        """
        Calculate the stiffness parameters for both notations.
        """
        C1111 = (1 - self.nu23) / (1 - self.nu23 - 2 * self.nu12 * self.nu21) * self.E1
        lam = (
            (self.nu12 * self.nu21 + self.nu23)
            / (1 - self.nu23 - 2 * self.nu12 * self.nu21)
            / (1 + self.nu23)
            * self.E2
        )
        self.stiffness66 = np.array(
            [
                [
                    C1111,
                    2 * self.nu12 * (lam + self.G23),
                    2 * self.nu12 * (lam + self.G23),
                    0,
                    0,
                    0,
                ],
                [2 * self.nu12 * (lam + self.G23), lam + 2 * self.G23, lam, 0, 0, 0],
                [2 * self.nu12 * (lam + self.G23), lam, lam + 2 * self.G23, 0, 0, 0],
                [0, 0, 0, 2 * self.G23, 0, 0],
                [0, 0, 0, 0, 2 * self.G12, 0],
                [0, 0, 0, 0, 0, 2 * self.G12],
            ]
        )
        self.stiffness3333 = self.mandel2tensor(self.stiffness66)


class Isotropy(TransverseIsotropy):
    """
    Isotropy class to express isotropic elasitc stiffness tensors.
    The class inherits from the Transverse Isotropy class.

    Parameters
    ----------
    E : float
        Young's modulus in Pa.
    nu : float
        Poisson's ratio (dimensionless).

    Attributes
    ----------
    E : float
        Young's modulus in Pa.
    nu : float
        Poisson's ratio (dimensionless).
    lam : float
        First Lamé constant in Pa.
    mu : float
        Second Lamé constant in Pa.
    """

    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.lam = self._get_lambda()
        self.mu = self._get_mu()
        super().__init__(self.E, self.E, self.mu, self.mu, self.nu)

    def _get_lambda(self):
        """
        Return the first Lamé constant from other material parameters.

        Returns
        -------
        float
            First Lamé constant in Pa.
        """
        return self.nu / (1 - 2 * self.nu) * 1 / (1 + self.nu) * self.E

    def _get_mu(self):
        """
        Return the second Lamé constant from other material parameters.

        Returns
        -------
        float
            Second Lamé constant in Pa.
        """
        return 1 / 2 * 1 / (1 + self.nu) * self.E
