# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:09:24 2022

@author: nicolas.christ@kit.edu

Mori-Tanaka Homogenization after Seelig (2016). Multi-inclusion implementation after Brylka (2017).
Eshelby Tensor is taken from Tandon, Weng (1984) but can also be found in Seelig (2016).
Halpin-Tsai homogenization after Fu, Lauke, Mai (2019, p. 143 ff.). Also, the effective planar stiffness 
matrix for the Halpin-Tsai homogenization is based on the laminate analogy approach after Fu, Lauke, Mai
(2019, p. 155 ff.).

Tested:
    - Young's modulus for "almost"sphere (a = 1) in correspondance to Isotropic implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from homopy.tensor import Tensor

sin = np.sin
cos = np.cos
tanh = np.tanh


class MoriTanaka(Tensor):
    """
    Mori Tanaka class to calculate the homogenized stiffness for fiber reinforced
    polymers with possibly different types of inclusions. The class inherits from
    the Tensor class.
    """

    def __init__(self, matrix, fiber, v_frac, a_ratio):
        """
        Initialize the object and call super class initialization.

        Parameters:
            - matrix : class object of the Elasticity class (or any child class)
                Polymer matrix material.
            - fiber : class object of the Elasticity class (or any child class)
                      or list of objects of the Elasticity class
                Fiber material.
            - v_frac : float
                Volume fraction of the fiber material within the matrix
                material.
            - a_ratio : float
                Aspect ratio of the fiber material.

        Object variables:
            - matrix : class object of the Elasticity class (or any child class)
                Polymer matrix material.
            - fiber : class object of the Elasticity class (or any child class)
                Fiber material.
            - Cm : ndarray of shape(6, 6)
                Stiffness of matrix material in normalized Voigt notation.
            - eye : ndarray of shape(6, 6)
                Identity tensor in normalized Voigt notation.

        Returns:
            - None
        """
        super().__init__()
        self.matrix = matrix
        self.fiber = fiber
        self.Cm = matrix.stiffness66
        self.eye = np.eye(6)

        # when the fiber parameter is a list, differnt types of inclusions are considered
        if not type(fiber) == list:
            self.fiber = fiber
            self.Cf = fiber.stiffness66
            self.v_frac = v_frac
            self.a_ratio = a_ratio
            self.eshelby66 = self.get_eshelby(self.a_ratio)

        else:
            assert (
                len(fiber) == len(v_frac) == len(a_ratio)
            ), "Dimensions of stiffnesses, v_fracs and a_ratios do not match!"
            self.nr_constituents = len(fiber)
            self.A_f_alpha = list()
            self.pol_alpha = list()
            self.c_alpha = (
                list()
            )  # vol_frac of phase alpha in reference to total inclusion vol_frac
            self.c_f = sum(v_frac)

            Cm_inv = np.linalg.inv(self.Cm)
            for i in range(self.nr_constituents):
                Cf_alpha = fiber[i].stiffness66
                pol = Cf_alpha - self.Cm
                self.pol_alpha.append(pol)
                S = self.get_eshelby(a_ratio[i])
                A_inv = self.eye + self.tensor_product(
                    S, self.tensor_product(Cm_inv, pol)
                )
                A = np.linalg.inv(A_inv)
                self.A_f_alpha.append(A)
                self.c_alpha.append(v_frac[i] / self.c_f)

    def get_eshelby(self, a_ratio, return_dim="66", shape="ellipsoid"):
        """
        Return the Eshelby tensor according to the fiber type.

        Parameters:
            - a_ratio : float
                Aspect ratio of fiber.
            - return_dim : string, default='66'
                Flag to determine whether the tensor should be returned in
                normalized Voigt or regular tensor notation (options: '66', '3333')
            - shape : string, default='ellipsoid'
                Flag to determine which assumptions are taken into consideration
                for the geometry of the fiber. So far not in use...

        Returns:
            - S : ndarray of shape(6, 6) or (3, 3, 3, 3)
                Eshelby inclusion tensor.

        """
        nu = self.matrix.nu
        a = a_ratio
        a2 = a**2
        g = a / (a2 - 1) ** (3 / 2) * (a * (a2 - 1) ** (1 / 2) - np.arccosh(a))
        S = np.zeros((3, 3, 3, 3))
        S[0, 0, 0, 0] = (
            1
            / (2 * (1 - nu))
            * (
                1
                - 2 * nu
                + (3 * a2 - 1) / (a2 - 1)
                - (1 - 2 * nu + 3 * a2 / (a2 - 1)) * g
            )
        )
        S[1, 1, 1, 1] = (
            3 / (8 * (1 - nu)) * a2 / (a2 - 1)
            + 1 / (4 * (1 - nu)) * (1 - 2 * nu - 9 / (4 * (a2 - 1))) * g
        )
        S[2, 2, 2, 2] = S[1, 1, 1, 1]
        S[1, 1, 2, 2] = (
            1
            / (4 * (1 - nu))
            * (a2 / (2 * (a2 - 1)) - (1 - 2 * nu + 3 / (4 * (a2 - 1))) * g)
        )
        S[2, 2, 1, 1] = S[1, 1, 2, 2]
        S[1, 1, 0, 0] = (
            -1 / (2 * (1 - nu)) * a2 / (a2 - 1)
            + 1 / (4 * (1 - nu)) * (3 * a2 / (a2 - 1) - (1 - 2 * nu)) * g
        )
        S[2, 2, 0, 0] = S[1, 1, 0, 0]
        S[0, 0, 1, 1] = (
            -1 / (2 * (1 - nu)) * (1 - 2 * nu + 1 / (a2 - 1))
            + 1 / (2 * (1 - nu)) * (1 - 2 * nu + 3 / (2 * (a2 - 1))) * g
        )
        S[0, 0, 2, 2] = S[0, 0, 1, 1]
        S[1, 2, 1, 2] = (
            1
            / (4 * (1 - nu))
            * (a2 / (2 * (a2 - 1)) + (1 - 2 * nu - 3 / (4 * (a2 - 1))) * g)
        )
        S[2, 1, 2, 1] = S[1, 2, 1, 2]
        S[0, 1, 0, 1] = (
            1
            / (4 * (1 - nu))
            * (
                1
                - 2 * nu
                - (a2 + 1) / (a2 - 1)
                - 1 / 2 * (1 - 2 * nu - 3 * (a2 + 1) / (a2 - 1)) * g
            )
        )
        S[0, 2, 0, 2] = S[0, 1, 0, 1]
        if return_dim == "66":
            return self.tensor2mandel(S)
        elif return_dim == "3333":
            return S

    def get_effective_stiffness(self):
        """
        Return the effective stiffness of a inhomogeneous material.

        Parameters:
            - None

        Returns:
            - C_eff : ndarray of shape (6, 6)
                Homogenized stiffness tensor in the normalized Voigt notation.
        """
        if not type(self.fiber) == list:
            Cm_inv = np.linalg.inv(self.Cm)
            pol = self.Cf - self.Cm
            A_inv = self.eye + self.tensor_product(
                self.eshelby66, self.tensor_product(Cm_inv, pol)
            )
            # A = np.linalg.inv(A_inv)
            C_eff = self.Cm + self.v_frac * self.tensor_product(
                pol, np.linalg.inv(self.v_frac * self.eye + (1 - self.v_frac) * A_inv)
            )
        else:
            pol_A_ave = np.zeros((6, 6))
            A_ave = np.zeros((6, 6))
            # calculating the averages
            for i in range(self.nr_constituents):
                A_ave += self.c_alpha[i] * self.A_f_alpha[i]
                pol_A_ave += self.c_alpha[i] * self.tensor_product(
                    self.pol_alpha[i], self.A_f_alpha[i]
                )

            C_eff = self.Cm + self.tensor_product(
                self.c_f * pol_A_ave,
                np.linalg.inv(self.c_f * A_ave + (1 - self.c_f) * self.eye),
            )

        return C_eff

    def get_average_stiffness(self, C_eff, N2, N4):
        """
        Return the averaged effective stiffness based on orientation tensors.

        Parameters:
            - C_eff : ndarray of shape(6, 6) or (3, 3, 3, 3)
                Effective stiffness in normalized Voigt or regular tensor notation.
            - N2 : ndarray of shape(3, 3)
                Orientation tensor of 2nd order.
            - N4 : ndarray of shape(3, 3, 3, 3)
                Orientation tensor of 4th order.

        Returns:
            - ... : ndarray of shape (6, 6)
                Averaged stiffness tensor in the normalized Voigt notation.
        """
        if C_eff.shape == (6, 6):
            C_eff = self.mandel2tensor(C_eff)

        b1 = (
            C_eff[0, 0, 0, 0]
            + C_eff[1, 1, 1, 1]
            - 2 * C_eff[0, 0, 1, 1]
            - 4 * C_eff[0, 1, 0, 1]
        )
        b2 = C_eff[0, 0, 1, 1] - C_eff[1, 1, 2, 2]
        b3 = C_eff[0, 1, 0, 1] + 1 / 2 * (C_eff[1, 1, 2, 2] - C_eff[1, 1, 1, 1])
        b4 = C_eff[1, 1, 2, 2]
        b5 = 1 / 2 * (C_eff[1, 1, 1, 1] - C_eff[1, 1, 2, 2])

        eye2 = np.eye(3)

        C_eff_ave = (
            b1 * N4
            + b2
            * (np.einsum("ij,kl->ijkl", N2, eye2) + np.einsum("ij,kl->ijkl", eye2, N2))
            + b3
            * (
                np.einsum("ik,lj->ijkl", N2, eye2)
                + np.einsum("ik,lj->ijlk", N2, eye2)
                + np.einsum("ik,lj->klij", eye2, N2)
                + np.einsum("ik,lj->ijlk", eye2, N2)
            )
            + b4 * (np.einsum("ij,kl->ijkl", eye2, eye2))
            + 2
            * b5
            * 1
            / 2
            * (
                np.einsum("ik,lj->ijkl", eye2, eye2)
                + np.einsum("ik,lj->ijlk", eye2, eye2)
            )
        )
        return self.tensor2mandel(C_eff_ave)

    def is_symmetric(self):
        """
        Print the symmetry status of the effective stiffness.
        """
        stiffness = self.get_effective_stiffness()
        # transform to Mandel notation
        stiffness[3:6, 0:3] *= np.sqrt(2)
        stiffness[0:3, 3:6] *= np.sqrt(2)
        stiffness[3:6, 3:6] *= 2
        stiffness = self.mandel2tensor(stiffness)
        left_minor = np.einsum("ijkl->jikl", stiffness)
        right_minor = np.einsum("ijkl->ijlk", stiffness)
        major = np.einsum("ijkl->klij", stiffness)
        if np.linalg.norm(stiffness - left_minor) < 1e-3:
            print("Left minor symmetry: passed")
        else:
            print("Left minor symmetry: failed")
            print(
                "The residuum was: res = {}".format(
                    np.linalg.norm(stiffness - left_minor)
                )
            )
        if np.linalg.norm(stiffness - right_minor) < 1e-3:
            print("Right minor symmetry: passed")
        else:
            print("Right minor symmetry: failed")
            print(
                "The residuum was: res = {}".format(
                    np.linalg.norm(stiffness - right_minor)
                )
            )
        if np.linalg.norm(stiffness - major) < 1e-3:
            print("Major symmetry: passed")
        else:
            print("Major symmetry: failed")
            print(
                "The residuum was: res = {}".format(np.linalg.norm(stiffness - major))
            )
        print("\n")


class HalpinTsai:
    def __init__(self, E_f, E_m, G_f, G_m, nu_f, nu_m, l_f, r_f, vol_f, package="hex"):
        """
        Class to perform the Halpin-Tsai homogenization.

        Parameters
        ----------
        E_f : Float
            Young's modulus of fiber.
        E_m : Float
            Young's modulus of matrix.
        G_f : Float
            Shear modulus of fiber.
        G_m : Float
            Shear modulus of matrix.
        nu_f : Float
            Poisson ratio of fiber.
        nu_m : Float
            Poisson ratio of matrix.
        l_f : Float
            Average length of fiber.
        r_f : Float
            Average radius of fiber.
        vol_f : Float
            Poisson ratio of matrix.
        package : String (default: hex), other options: square
            Package structure of fibers in composite.
        """
        self.E_f = E_f
        self.E_m = E_m
        self.G_f = G_f
        self.G_m = G_m
        self.nu_f = nu_f
        self.nu_m = nu_m
        self.l_f = l_f
        self.r_f = r_f
        self.vol_f = vol_f
        self.package = package
        self.get_effective_parameters()

    def get_effective_parameters(self):
        """
        Calculates the effective parameters of a single lamina for given constituent parameters.

        Raises
        ------
        ValueError
            Package can only be "hex" or "square".

        Returns
        -------
        None.
        """
        if self.package != "hex" and self.package != "square":
            raise ValueError('Package must be either "hex" or "square"!')

        if self.package == "hex":
            p = 1 / 2 * np.log(2 * np.pi / (np.sqrt(3) * self.vol_f))
        else:
            p = 1 / 2 * np.log(np.pi / self.vol_f)

        beta = np.sqrt(2 * np.pi * self.G_m / (self.E_f * (np.pi * self.r_f**2) * p))
        nu1 = (self.E_f / self.E_m - 1) / (self.E_f / self.E_m + 2)
        nu2 = (self.G_f / self.G_m - 1) / (self.G_f / self.G_m + 1)

        self.E11 = self.E_f * (
            1 - tanh(beta * self.l_f / 2) / (beta * self.l_f / 2)
        ) * self.vol_f + self.E_m * (1 - self.vol_f)
        self.E22 = self.E_m * (1 + 2 * nu1 * self.vol_f) / (1 - nu1 * self.vol_f)
        self.G12 = self.G_m * (1 + 2 * nu2 * self.vol_f) / (1 - nu2 * self.vol_f)
        self.nu12 = self.nu_f * self.vol_f + self.nu_m * (1 - self.vol_f)
        self.nu21 = self.nu12 * self.E22 / self.E11

    def get_stiffness(self):
        """
        Return the planar stiffness based on the effective parameters of a single lamina.

        Returns
        -------
        C : ndarray of shape(3,3)
            Planar stiffness of lamina.
        """
        Q11 = self.E11 / (1 - self.nu12 * self.nu21)
        Q12 = self.nu21 * Q11
        Q16 = 0
        Q22 = self.E22 / (1 - self.nu12 * self.nu21)
        Q26 = 0
        Q66 = self.G12
        C = np.array([[Q11, Q12, Q16], [Q12, Q22, Q26], [Q16, Q26, Q66]])
        return C


class Laminate:
    def __init__(self, lamina_stiffnesses, angles, vol_fracs=None):
        """
        Class to average over n laminas from Halpin-Tsai homogenization.

        Parameters
        ----------
        lamina_stiffnesses : array of shape (n,)
            Individual stiffness of n laminas.
        angles : array of shape (n,)
            Individual angle of ith lamina in radians.
        vol_fracs : array of shape (n,)
            Volume fraction of ith lamina (must sum to 1). If None is given,
            each lamina is averaged equally.
        """
        self.lamina_stiffnesses = lamina_stiffnesses
        self.angles = angles
        if not vol_fracs is None:
            assert (
                len(lamina_stiffnesses) == len(angles) == len(vol_fracs)
            ), "Dimensions of lamina_stiffnesses, angles and vol_fracs do not match!"
            self.vol_fracs = vol_fracs
        else:
            self.vol_fracs = (
                1 / len(lamina_stiffnesses) * np.ones(len(lamina_stiffnesses))
            )

    def get_effective_stiffness(self):
        """
        Return effective stiffness of laminate.

        Returns
        -------
        C_eff : ndarray of shape(3,3)
            Effective stiffness of laminate.
        """
        C_eff_temp = np.zeros(6)
        for i in range(len(self.lamina_stiffnesses)):
            # rotate by angle
            Q_temp = self.rotate_stiffness(self.lamina_stiffnesses[i], self.angles[i])
            C_eff_temp += self.vol_fracs[i] * Q_temp
        C_eff = np.array(
            [
                [C_eff_temp[0], C_eff_temp[2], C_eff_temp[4]],
                [C_eff_temp[2], C_eff_temp[1], C_eff_temp[5]],
                [C_eff_temp[4], C_eff_temp[5], C_eff_temp[3]],
            ]
        )
        return C_eff

    @staticmethod
    def rotate_stiffness(lamina_stiffness, angle):
        """
        Return planarly rotated stiffness matrix.

        Parameters
        ----------
        lamina_stiffness : ndarray of shape(3, 3)
            Stiffness matrix of lamina.
        angle : float
            Planar angle to rotate the stiffness matrix about.

        Returns
        -------
        rot_stiffness : ndarray of shape(3, 3)
            Rotated stiffness matrix.
        """
        m = cos(angle)
        n = sin(angle)
        rot_mat = np.array(
            [
                [m**4, n**4, 2 * m**2 * n**2, 4 * m**2 * n**2],
                [n**4, m**4, 2 * m**2 * n**2, 4 * m**2 * n**2],
                [
                    m**2 * n**2,
                    m**2 * n**2,
                    m**4 + n**4,
                    -4 * m**2 * n**2,
                ],
                [
                    m**2 * n**2,
                    m**2 * n**2,
                    -2 * m**2 * n**2,
                    (m**2 - n**2) ** 2,
                ],
                [
                    m**3 * n,
                    -m * n**3,
                    m * n**3 - m**3 * n,
                    2 * (m * n**3 - m**3 * n),
                ],
                [
                    m * n**3,
                    -(m**3) * n,
                    m**3 * n - m * n**3,
                    2 * (m**3 * n - m * n**3),
                ],
            ]
        )
        Q = lamina_stiffness
        flat_stiffness = np.array([Q[0, 0], Q[1, 1], Q[0, 1], Q[2, 2]])
        rot_stiffness = np.einsum("ij,j->i", rot_mat, flat_stiffness)
        return rot_stiffness
