# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:09:24 2022

@author: nicolas.christ@kit.edu

Mori-Tanaka Homogenization after [Benveniste1987]_. Multi-inclusion implementation after [Brylka2017]_.
Eshelby's tensor is taken from [Tandon1984]_ but can also be found in [Gross2016]_. Thoroughly literature
on Eshelby's tensor can also be found in [Mura1987]_ (pp. 74 ff.).
Halpin-Tsai homogenization after [Fu2019]_ (pp. 143 ff.). Also, the effective planar stiffness
matrix for the Halpin-Tsai homogenization is based on the laminate analogy approach after [Fu2019]_
(pp. 155 ff.).
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

    Parameters
    ----------
    matrix : Elasticity
        Polymer matrix material in normalized Voigt notation.
    fiber : Elasticity or list of Elasticity
        Fiber material in normalized Voigt notation.
    v_frac : float
        Volume fraction of the fiber material within the matrix
        material.
    a_ratio : float or list of floats
        Aspect ratio of the fiber material.
    shape : string or list of strings, default='ellipsoid'
        Flag to determine which assumptions are taken into consideration
        for the geometry of the fiber (options: 'ellipsoid', 'sphere', 'needle')
    N4 : ndarray or list of ndarrays of shape (3, 3, 3, 3), default=None
        Orientation tensor(s) of 4th order.
    symmetrize : boolean, default='False'
        Flag to determine whether the effective and orientation averaged stiffnesses shall
        be symmetrized. For this the method in [Segura2023]_ is used.

    Attributes
    ----------
    matrix : Elasticity
        Polymer matrix material.
    fiber : Elasticity or list of Elasticity
        Fiber (or other inclusion) materials.
    Cm : ndarray of shape (6, 6)
        Stiffness of matrix material in normalized Voigt notation in Pa.
    eye : ndarray of shape (6, 6)
        Identity tensor in normalized Voigt notation.
    N2 : ndarray or list of ndarrays of shape (3, 3)
        Orientation tensor(s) of 2nd order.
    N4 : ndarray or list of ndarrays of shape (3, 3, 3, 3)
        Orientation tensor(s) of 4th order.
    effective_stiffness3333 : ndarray of shape (3, 3, 3, 3)
        Holds the stiffness values in the regular tensor notation in Pa.
        When orientations are given, these are included directly.
    effective_stiffness66 : ndarray of shape (6, 6)
        Holds the stiffness values in the normalized Voigt notation in Pa.
        When orientations are given, these are included directly.
    """

    def __init__(
        self,
        matrix,
        fiber,
        v_frac,
        a_ratio,
        shape="ellipsoid",
        N4=None,
        symmetrize=False,
    ):

        super().__init__()
        self.matrix = matrix
        self.fiber = fiber if isinstance(fiber, list) else [fiber]
        self.v_frac = v_frac if isinstance(v_frac, list) else [v_frac]
        self.a_ratio = a_ratio if isinstance(a_ratio, list) else [a_ratio]
        self.shape = shape if isinstance(shape, list) else [shape]

        self.symmetrize = symmetrize

        self.Cm = matrix.stiffness66
        self.eye = np.eye(6)  # make symmetric identity

        assert (
            len(self.fiber) == len(self.v_frac) == len(self.a_ratio)
        ), "Length of stiffnesses, v_fracs, and a_ratios do not match!"

        self.nr_constituents = len(self.fiber)
        self.A_f_alpha = list()
        self.stiff_diff = list()
        self.c_alpha = (
            list()
        )  # vol_frac of phase alpha in reference to total inclusion vol_frac
        self.c_f = sum(self.v_frac)

        Cm_inv = np.linalg.inv(self.Cm)

        self.N2 = list()
        self.N4 = list()

        for i in range(self.nr_constituents):
            if N4 is None:
                N4_tmp = np.zeros((3, 3, 3, 3))
                N4_tmp[0, 0, 0, 0] = 1
            else:
                N4 = N4 if isinstance(N4, list) else [N4]
                assert len(self.fiber) == len(
                    N4
                ), "If orientation tensors are given, it must be one for each inclusion type!"
                N4_tmp = N4[i]

            N2 = np.einsum("ijkk->ij", N4_tmp)
            self.N2.append(N2)
            self.N4.append(N4_tmp)

            Cf_alpha = self.fiber[i].stiffness66
            stiff_diff = Cf_alpha - self.Cm
            self.stiff_diff.append(stiff_diff)

            if len(self.shape) == 1:
                self.shape = self.nr_constituents * self.shape
            assert len(self.fiber) == len(
                self.shape
            ), "When `shape` is a list, it must have the same length as `fiber`!"

            S = self._get_eshelby(self.a_ratio[i], return_dim="66", shape=self.shape[i])

            A_inv = self.eye + self.tensor_product(
                S, self.tensor_product(Cm_inv, self.stiff_diff[i])
            )
            A = np.linalg.inv(A_inv)
            self.A_f_alpha.append(A)
            self.c_alpha.append(self.v_frac[i] / self.c_f)

        self.effective_stiffness66 = self.get_effective_stiffness()
        self.effective_stiffness3333 = self.mandel2tensor(self.effective_stiffness66)

    def _get_eshelby(self, a_ratio, return_dim="66", shape="ellipsoid"):
        """
        Return the Eshelby tensor according to the fiber type.

        Parameters
        ----------
        a_ratio : float
            Aspect ratio of fiber (dimensionless).
        return_dim : string, default='66'
            Flag to determine whether the tensor should be returned in
            normalized Voigt or regular tensor notation (options: '66', '3333')
        shape : string, default='ellipsoid'
            Flag to determine which assumptions are taken into consideration
            for the geometry of the fiber (options: 'ellipsoid', 'sphere', 'needle')

        Returns
        -------
        S : ndarray of shape (6, 6) or (3, 3, 3, 3)
            Eshelby inclusion tensor.
        """

        nu = self.matrix.nu
        a = a_ratio
        a2 = a**2
        S = np.zeros((3, 3, 3, 3))
        if shape == "ellipsoid":
            g = a / (a2 - 1) ** (3 / 2) * (a * (a2 - 1) ** (1 / 2) - np.arccosh(a))
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
            S[1, 1, 1, 1] = S[2, 2, 2, 2] = (
                3 / (8 * (1 - nu)) * a2 / (a2 - 1)
                + 1 / (4 * (1 - nu)) * (1 - 2 * nu - 9 / (4 * (a2 - 1))) * g
            )
            S[1, 1, 2, 2] = S[2, 2, 1, 1] = (
                1
                / (4 * (1 - nu))
                * (a2 / (2 * (a2 - 1)) - (1 - 2 * nu + 3 / (4 * (a2 - 1))) * g)
            )
            S[1, 1, 0, 0] = S[2, 2, 0, 0] = (
                -1 / (2 * (1 - nu)) * a2 / (a2 - 1)
                + 1 / (4 * (1 - nu)) * (3 * a2 / (a2 - 1) - (1 - 2 * nu)) * g
            )
            S[0, 0, 1, 1] = S[0, 0, 2, 2] = (
                -1 / (2 * (1 - nu)) * (1 - 2 * nu + 1 / (a2 - 1))
                + 1 / (2 * (1 - nu)) * (1 - 2 * nu + 3 / (2 * (a2 - 1))) * g
            )
            S[1, 2, 1, 2] = S[2, 1, 2, 1] = S[2, 1, 1, 2] = S[1, 2, 2, 1] = (
                1
                / (4 * (1 - nu))
                * (a2 / (2 * (a2 - 1)) + (1 - 2 * nu - 3 / (4 * (a2 - 1))) * g)
            )
            S[0, 1, 0, 1] = S[0, 2, 0, 2] = S[1, 0, 1, 0] = S[1, 0, 0, 1] = S[
                0, 1, 1, 0
            ] = S[2, 0, 2, 0] = S[2, 0, 0, 2] = S[0, 2, 2, 0] = (
                1
                / (4 * (1 - nu))
                * (
                    1
                    - 2 * nu
                    - (a2 + 1) / (a2 - 1)
                    - 1 / 2 * (1 - 2 * nu - 3 * (a2 + 1) / (a2 - 1)) * g
                )
            )
        elif shape == "sphere":
            fac1 = 15 * (1 - nu)
            eye3 = np.eye(3)

            S = (
                1
                / fac1
                * (
                    (5 * nu - 1) * np.einsum("ij,kl->ijkl", eye3, eye3)
                    + (4 - 5 * nu)
                    * (
                        np.einsum("ik,jl->ijkl", eye3, eye3)
                        + np.einsum("il,jk->ijkl", eye3, eye3)
                    )
                )
            )

        elif shape == "needle":
            # Here the aspect ratio a describes the relation between the two minor axes
            pre_fac = 1 / (2 * (1 - nu))
            fac1 = 1 / (a + 1)
            fac2 = a / (a + 1)
            fac3 = 1 - 2 * nu
            fac4 = (a + 1) ** 2

            S[1, 1, 1, 1] = pre_fac * ((1 + 2 * a) / fac4 + fac3 * fac1)
            S[2, 2, 2, 2] = pre_fac * ((a2 + 2 * a) / fac4 + fac3 * fac2)
            S[1, 1, 2, 2] = pre_fac * (1 / fac4 - fac3 * fac1)
            S[2, 2, 0, 0] = pre_fac * 2 * nu * fac2
            S[1, 1, 0, 0] = pre_fac * 2 * nu * fac1
            S[2, 2, 1, 1] = pre_fac * (a2 / fac4 - fac3 * fac2)
            S[2, 1, 2, 1] = S[1, 2, 1, 2] = S[1, 2, 2, 1] = S[2, 1, 1, 2] = pre_fac * (
                (a2 + 1) / (2 * fac4) + fac3 / 2
            )
            S[2, 0, 2, 0] = S[0, 2, 0, 2] = S[0, 2, 2, 0] = S[2, 0, 0, 2] = 1 / 2 * fac2
            S[0, 1, 0, 1] = S[1, 0, 1, 0] = S[1, 0, 0, 1] = S[0, 1, 1, 0] = 1 / 2 * fac1
        else:
            raise ValueError(
                "Please chose a valid 'shape' option. "
                "Options supported: 'ellipsoid', 'sphere', 'needle'."
            )

        if return_dim == "66":
            return self.tensor2mandel(S)
        elif return_dim == "3333":
            return S

    def get_effective_stiffness(self):
        """
        Return the effective stiffness of the composite material, based on Eq. 14a in [Benveniste1987]_.

        Returns
        -------
        C_eff : ndarray of shape (6, 6)
            Homogenized stiffness tensor in the normalized Voigt notation in Pa.
        """

        pol_A_ave = np.zeros((6, 6))
        A_ave = np.zeros((6, 6))
        # calculating the averages
        for i in range(self.nr_constituents):
            # here comes the orientation average

            # weight by stiff_diff...
            weighted_A_f_alpha_66 = self.tensor_product(
                self.stiff_diff[i], self.A_f_alpha[i]
            )
            weighted_A_f_alpha_3333 = self.mandel2tensor(weighted_A_f_alpha_66)
            # orientation average...
            ave_weighted_A_f_alpha_3333 = self.get_orientation_average(
                weighted_A_f_alpha_3333, self.N2[i], self.N4[i]
            )
            ave_weighted_A_f_alpha_66 = self.tensor2mandel(ave_weighted_A_f_alpha_3333)
            # remove weight by inverse...
            ave_A_f_alpha = self.tensor_product(
                np.linalg.inv(self.stiff_diff[i]), ave_weighted_A_f_alpha_66
            )

            A_ave += self.c_alpha[i] * ave_A_f_alpha
            pol_A_ave += self.c_alpha[i] * self.tensor_product(
                self.stiff_diff[i], ave_A_f_alpha
            )

        pol_A_ave_inv = np.linalg.inv(pol_A_ave)
        X = (1 - self.c_f) * pol_A_ave_inv
        Y = self.c_f * self.tensor_product(A_ave, pol_A_ave_inv)
        Y_symm = 1 / 2 * (Y + Y.T)

        if self.symmetrize == False:
            C_eff = self.Cm + self.c_f * np.linalg.inv(X + Y)
        else:
            C_eff = self.Cm + self.c_f * np.linalg.inv(X + Y_symm)

        return C_eff

    def get_average_stiffness(self, N4, return_dim="66"):
        """
        Return the averaged effective stiffness based on orientation tensors. Overwrites the
        object variables self.effective_stiffness66 and self.effective_stiffness3333.

        Parameters
        ----------
        N4 : ndarray or list of ndarrays of shape (3, 3, 3, 3)
            Orientation tensor of 4th order.
        return_dim : string, default='66'
            Flag to determine whether the tensor should be returned in
            normalized Voigt or regular tensor notation (options: '66', '3333')

        Returns
        -------
        ndarray of shape (6, 6) or (3, 3, 3, 3)
            Averaged stiffness tensor in normalized Voigt or regular tensor notation in Pa.
        """

        # overwrite N2 and N4
        N4 = N4 if isinstance(N4, list) else [N4]
        assert len(self.fiber) == len(
            N4
        ), "One orientation tensor for each inclusion type!"

        self.N2 = list()
        self.N4 = list()
        for i in range(self.nr_constituents):
            N4_tmp = N4[i]
            N2 = np.einsum("ijkk->ij", N4_tmp)
            self.N2.append(N2)
            self.N4.append(N4_tmp)

        self.effective_stiffness66 = self.get_effective_stiffness()
        self.effective_stiffness3333 = self.mandel2tensor(self.effective_stiffness66)

        if return_dim == "66":
            return self.effective_stiffness66
        else:
            return self.effective_stiffness66

    @staticmethod
    def get_orientation_average(tensor, N2, N4):
        """
        Return the orientation average of a tensor after [Advani1987]_, Eq. 29.

        Parameters
        ----------
        tensor : ndarray of shape (3, 3, 3, 3)
            Tensor to be averaged (must be transversely isotropic).
        N2 : ndarray of shape (3, 3)
            Orientation tensor of 2nd order.
        N4 : ndarray of shape (3, 3, 3, 3)
            Orientation tensor of 4th order.

        Returns
        -------
        ave_tensor : ndarray of shape (3, 3, 3, 3)
            Orientation average of given tensor.
        """

        if N4.shape == (6, 6):
            N4 = Tensor().mandel2tensor(N4)

        b1 = (
            tensor[0, 0, 0, 0]
            + tensor[1, 1, 1, 1]
            - 2 * tensor[0, 0, 1, 1]
            - 4 * tensor[0, 1, 0, 1]
        )
        b2 = tensor[0, 0, 1, 1] - tensor[1, 1, 2, 2]
        b3 = tensor[0, 1, 0, 1] + 1 / 2 * (tensor[1, 1, 2, 2] - tensor[1, 1, 1, 1])
        b4 = tensor[1, 1, 2, 2]
        b5 = 1 / 2 * (tensor[1, 1, 1, 1] - tensor[1, 1, 2, 2])

        eye3 = np.eye(3)

        ave_tensor = (
            b1 * N4
            + b2
            * (np.einsum("ij,kl->ijkl", N2, eye3) + np.einsum("ij,kl->ijkl", eye3, N2))
            + b3
            * (
                np.einsum("ik,lj->ijkl", N2, eye3)
                + np.einsum("ik,lj->ijlk", N2, eye3)
                + np.einsum("ik,lj->klij", eye3, N2)
                + np.einsum("ik,lj->ijlk", eye3, N2)
            )
            + b4 * (np.einsum("ij,kl->ijkl", eye3, eye3))
            + 2
            * b5
            * 1
            / 2
            * (
                np.einsum("ik,lj->ijkl", eye3, eye3)
                + np.einsum("ik,lj->ijlk", eye3, eye3)
            )
        )

        return ave_tensor

    def is_symmetric(self):
        """
        Print the symmetry status of the effective stiffness.
        """

        left_minor = np.einsum("ijkl->jikl", self.effective_stiffness3333)
        right_minor = np.einsum("ijkl->ijlk", self.effective_stiffness3333)
        major = np.einsum("ijkl->klij", self.effective_stiffness3333)
        if np.allclose(self.effective_stiffness3333, left_minor, rtol=1e-6):
            print("Left minor symmetry: passed")
        else:
            print("Left minor symmetry: failed")
            print(
                "The rel. residuum for left minor sym. is: res = {}".format(
                    np.linalg.norm(self.effective_stiffness3333 - left_minor)
                    / np.linalg.norm(self.effective_stiffness3333)
                )
            )
        if np.allclose(self.effective_stiffness3333, right_minor, rtol=1e-6):
            print("Right minor symmetry: passed")
        else:
            print("Right minor symmetry: failed")
            print(
                "The rel. residuum for right minor sym. is: res = {}".format(
                    np.linalg.norm(self.effective_stiffness3333 - right_minor)
                    / np.linalg.norm(self.effective_stiffness3333)
                )
            )
        if np.allclose(self.effective_stiffness3333, major, rtol=1e-6):
            print("Major symmetry: passed")
        else:
            print("Major symmetry: failed")
            print(
                "The rel. residuum for major sym. is: res = {}".format(
                    np.linalg.norm(self.effective_stiffness3333 - major)
                    / np.linalg.norm(self.effective_stiffness3333)
                )
            )
        print("\n")


class HalpinTsai:
    """
    Halpin Tsai class to calculate the homogenized stiffness for fiber reinforced
    polymers as laminas. This is then used as input for the Laminate class.

    Parameters
    ----------
    E_f : float
        Young's modulus of fiber in Pa.
    E_m : float
        Young's modulus of matrix in Pa.
    G_f : float
        Shear modulus of fiber in Pa.
    G_m : float
        Shear modulus of matrix in Pa.
    nu_f : float
        Poisson ratio of fiber (dimensionless).
    nu_m : float
        Poisson ratio of matrix (dimensionless).
    l_f : float
        Average length of fiber in m.
    r_f : float
        Average radius of fiber in m.
    vol_f : float
        Poisson ratio of matrix (dimensionless).
    package : string, default: hex
        Package structure of fibers in composite (options: 'hex', 'square').

    Attributes
    ----------
    E_f : float
        Young's modulus of fiber in Pa.
    E_m : float
        Young's modulus of matrix in Pa.
    G_f : float
        Shear modulus of fiber in Pa.
    G_m : float
        Shear modulus of matrix in Pa.
    nu_f : float
        Poisson ratio of fiber (dimensionless).
    nu_m : float
        Poisson ratio of matrix (dimensionless).
    l_f : float
        Average length of fiber in m.
    r_f : float
        Average radius of fiber in m.
    vol_f : float
        Poisson ratio of matrix (dimensionless).
    package : string, default: hex
        Package structure of fibers in composite (options: 'hex', 'square').
    effective_stiffness33 : ndarray of shape (3, 3)
        Holds the stiffness values in the reduced, normalized Voigt notation in Pa.
    """

    def __init__(self, E_f, E_m, G_f, G_m, nu_f, nu_m, l_f, r_f, vol_f, package="hex"):

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
        self._get_effective_parameters()
        self.effective_stiffness33 = self.get_effective_stiffness()

    def _get_effective_parameters(self):
        """
        Calculates the effective parameters of a single lamina for given constituent parameters.

        Raises
        ------
        ValueError
            Package can only be "hex" or "square".
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

    def get_effective_stiffness(self):
        """
        Return the planar stiffness based on the effective parameters of a single lamina.

        Returns
        -------
        C : ndarray of shape (3, 3)
            Planar stiffness of lamina.
        """
        fac = 2  # necessary to stay in normalized Voigt notation

        Q11 = self.E11 / (1 - self.nu12 * self.nu21)
        Q12 = self.nu21 * Q11
        Q16 = 0
        Q22 = self.E22 / (1 - self.nu12 * self.nu21)
        Q26 = 0
        Q66 = fac * self.G12
        C = np.array([[Q11, Q12, Q16], [Q12, Q22, Q26], [Q16, Q26, Q66]])
        return C


class Laminate:
    """
    Class to average over n laminas from Halpin-Tsai homogenization.

    Parameters
    ----------
    lamina_stiffnesses : array of shape (n,)
        Individual stiffness of n laminas in Pa.
    angles : array of shape (n,)
        Individual angle of ith lamina in radians.
    vol_fracs : array of shape (n,)
        Volume fraction of ith lamina (must sum to 1). If None is given,
        each lamina is averaged equally.

    Attributes
    ----------
    effective_stiffness33 : ndarray of shape (3, 3)
        Holds the stiffness values in the reduced, normalized Voigt notation in Pa.
    """

    def __init__(self, lamina_stiffnesses, angles, vol_fracs=None):

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
        self.effective_stiffness33 = self.get_effective_stiffness()

    def get_effective_stiffness(self):
        """
        Return effective stiffness of laminate.

        Returns
        -------
        C_eff : ndarray of shape (3, 3)
            Effective stiffness of laminate in Pa.
        """
        C_eff = np.zeros((3, 3))
        for i in range(len(self.lamina_stiffnesses)):
            # rotate by angle
            Q_temp = self.rotate_stiffness(self.lamina_stiffnesses[i], self.angles[i])
            C_eff += self.vol_fracs[i] * Q_temp

        return C_eff

    @staticmethod
    def rotate_stiffness(lamina_stiffness, angle):
        r"""
        Return planarly rotated stiffness matrix. The planar rotation matrix around the z-axis
        has the form

        .. math::
                \underline{R}=\begin{pmatrix}
                \cos\phi & -\sin\phi & 0\\
                \sin\phi &  \cos\phi& 0\\
                0 & 0 & 1
                \end{pmatrix},

        from which the transformation matrix was extracted in accordance to [Slawinksi2010]_,
        Eq. 5.2.19. 

        Parameters
        ----------
        lamina_stiffness : ndarray of shape (3, 3)
            Stiffness matrix of lamina in Pa.
        angle : float
            Planar angle to rotate the stiffness matrix about in radiants.

        Returns
        -------
        rot_stiffness : ndarray of shape (3, 3)
            Rotated stiffness matrix in Pa.
        """
        m = cos(angle)
        n = sin(angle)

        b = np.sqrt(2)
        R = np.array(
            [
                [m**2, n**2, b * (m * n)],
                [n**2, m**2, -b * (m * n)],
                [
                    -b * m * n,
                    b * m * n,
                    (m**2 - n**2),
                ],
            ]
        )
        R_inv = R.T
        Q = lamina_stiffness.copy()

        rot_stiffness = np.einsum("ij,jk,kl->il", R_inv, Q, R)

        return rot_stiffness
