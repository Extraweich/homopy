# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:09:24 2022

@author: nicolas.christ@kit.edu

Module that contains the linear elastic stiffness classes of Isotropy and Transverse Isotropy.
"""

import numpy as np
from .tensor import Tensor


class Elasticity(Tensor):
    r"""
    Elasticity class to express generic elasitc stiffness tensors. 
    The class inherits from the Tensor class. 

    The generic stiffness matrix has the following form in the 
    normalized Voigt (Mandel) notation

    .. math::
            \underline{\underline{C}} = \begin{pmatrix}
                                        C_{1111} & C_{1122} & C_{1133} & \sqrt{2}C_{1123} & \sqrt{2}C_{1131} & \sqrt{2}C_{1112} \\
                                        C_{2211} & C_{2222} & C_{2233} & \sqrt{2}C_{2223} & \sqrt{2}C_{2231} & \sqrt{2}C_{2212} \\
                                        C_{3311} & C_{3322} & C_{3333} & \sqrt{2}C_{3323} & \sqrt{2}C_{3331} & \sqrt{2}C_{3312} \\
                                        \sqrt{2}C_{2311} & \sqrt{2}C_{2322} & \sqrt{2}C_{2333} & 2 C_{2323} & 2 C_{2331} & 2 C_{2312} \\
                                        \sqrt{2}C_{3111} & \sqrt{2}C_{3122} & \sqrt{2}C_{3133} & 2 C_{3123} & 2 C_{3131} & 2 C_{3112} \\
                                        \sqrt{2}C_{1211} & \sqrt{2}C_{1222} & \sqrt{2}C_{1233} & 2 C_{1223} & 2 C_{1231} & 2 C_{1212} \\
                                        \end{pmatrix},

    which is a symmetric matrix, giving 21 parameters.

    Attributes
    ----------
    stiffness3333 : ndarray of shape (3, 3, 3, 3)
        Stiffness values in the regular tensor notation.
    stiffness66 : ndarray of shape (6, 6)
        Stiffness values in the normalized Voigt notation.

    """

    def __init__(self):
        super().__init__()
        self.stiffness3333 = np.zeros((3, 3, 3, 3))
        self.stiffness66 = np.zeros((6, 6))


class Orthotropy(Elasticity):
    r"""
    Orthotropy class to express orthotropic elastic stiffness 
    tensors. The class inherits from the Elasticity class.

    The orthotropic stiffness matrix has the following form 
    in the normalized Voigt (Mandel) notation

    .. math::
            \underline{\underline{C}} = \begin{pmatrix}
                                            \frac{1-\nu_{23}\nu_{32}}{D}E_{1}&
                                            \frac{\nu_{21}+\nu_{23}\nu_{31}}{D}E_{1}&
                                            \frac{\nu_{31}+\nu_{32}\nu_{21}}{D}E_{1}
                                            \\
                                            \frac{\nu_{12}+\nu_{13}\nu_{32}}{D}E_{2} &
                                            \frac{1-\nu_{13}\nu_{31}}{D}E_{2}&
                                            \frac{\nu_{32}+\nu_{31}\nu_{12}}{D}E_{2}
                                            \\
                                            \frac{\nu_{13}+\nu_{12}\nu_{23}}{D}E_{3} &
                                            \frac{\nu_{23}+\nu_{21}\nu_{13}}{D}E_{3} &
                                            \frac{1-\nu_{12}\nu_{21}}{D}E_{3}
                                            \\
                                            & & &2G_{23}
                                            \\
                                            & & & &2G_{13}
                                            \\
                                            & & & & &2G_{12}
                                        \end{pmatrix},

    where

    .. math::
        D=1-\nu_{12}\nu_{21}-\nu_{13}\nu_{31}-\nu_{23}\nu_{32}-2\nu_{12}\nu_{23}\nu_{31}.

    The corresponding compliance matrix is

    .. math::
            \underline{\underline{S}} = \begin{pmatrix}
                                            \frac{1}{E_1} & -\frac{\nu_{21}}{E_2} & -\frac{\nu_{31}}{E_3} & & & \\
                                            -\frac{\nu_{12}}{E_1} & \frac{1}{E_2} & -\frac{\nu_{32}}{E_3} & & & \\
                                            -\frac{\nu_{13}}{E_1} & -\frac{\nu_{23}}{E_2} & \frac{1}{E_3} & & & \\
                                            & & & \frac{1}{2G_{23}} & & \\
                                            & & & & \frac{1}{2G_{13}} & \\
                                            & & & & & \frac{1}{2G_{12}}
                                        \end{pmatrix}.

    Parameters
    ----------
    E1 : float
        Young's modulus in first principal direction.
    E2 : float
        Young's modulus in second principal direction.
    E3 : float
        Young's modulus in third principal direction.
    G12 : float
        Shear modulus in the first-second plane.
    G13 : float
        Shear modulus in the first-third plane.
    G23 : float
        Shear modulus in the second-third plane.
    nu12 : float
        Poisson's ratio to express strain in second principal direction caused by load in first principal direction (dimensionless).
    nu13 : float
        Poisson's ratio to express strain in third principal direction caused by load in first principal direction (dimensionless).
    nu23 : float
        Poisson's ratio to express strain in third principal direction caused by load in second principal direction (dimensionless).

    Attributes
    ----------
    E1 : float
        Young's modulus in first principal direction.
    E2 : float
        Young's modulus in second principal direction.
    E3 : float
        Young's modulus in third principal direction.
    G12 : float
        Shear modulus in the first-second plane.
    G13 : float
        Shear modulus in the first-third plane.
    G23 : float
        Shear modulus in the second-third plane.
    nu12 : float
        Poisson's ratio to express strain in second principal direction caused by load in first principal direction (dimensionless).
    nu21 : float
        Poisson's ratio to express strain in first principal direction caused by load in second principal direction (dimensionless).
    nu13 : float
        Poisson's ratio to express strain in third principal direction caused by load in first principal direction (dimensionless).
    nu31 : float
        Poisson's ratio to express strain in third principal direction caused by load in first principal direction (dimensionless).
    nu23 : float
        Poisson's ratio to express strain in third principal direction caused by load in second principal direction (dimensionless).
    nu32 : float
        Poisson's ratio to express strain in third principal direction caused by load in second principal direction (dimensionless).

    """

    def __init__(self, E1, E2, E3, G12, G13, G23, nu12, nu13, nu23):
        super().__init__()
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.G12 = G12
        self.G13 = G13
        self.G23 = G23
        self.nu12 = nu12
        self.nu21 = nu12 * E2 / E1
        self.nu13 = nu13
        self.nu31 = nu13 * E3 / E1
        self.nu23 = nu23
        self.nu32 = nu23 * E3 / E2
        self._get_stiffness()

    def _get_stiffness(self):
        """
        Calculate the stiffness parameters for both notations.
        """
        D = (
            1
            - self.nu12 * self.nu21
            - self.nu13 * self.nu31
            - self.nu23 * self.nu32
            - 2 * self.nu12 * self.nu23 * self.nu31
        )

        self.stiffness66 = np.array(
            [
                [
                    (1 - self.nu23 * self.nu32) / D * self.E1,
                    (self.nu21 + self.nu23 * self.nu31) / D * self.E1,
                    (self.nu31 + self.nu32 * self.nu21) / D * self.E1,
                    0,
                    0,
                    0,
                ],
                [
                    (self.nu12 + self.nu13 * self.nu32) / D * self.E2,
                    (1 - self.nu13 * self.nu31) / D * self.E2,
                    (self.nu32 + self.nu31 * self.nu12) / D * self.E2,
                    0,
                    0,
                    0,
                ],
                [
                    (self.nu13 + self.nu12 * self.nu23) / D * self.E3,
                    (self.nu23 + self.nu21 * self.nu13) / D * self.E3,
                    (1 - self.nu12 * self.nu21) / D * self.E3,
                    0,
                    0,
                    0,
                ],
                [0, 0, 0, 2 * self.G23, 0, 0],
                [0, 0, 0, 0, 2 * self.G12, 0],
                [0, 0, 0, 0, 0, 2 * self.G12],
            ]
        )
        self.stiffness3333 = self.mandel2tensor(self.stiffness66)


class TransverseIsotropy(Elasticity):
    r"""
    Transverse Isotropy class to express transverse-isotropic 
    elasitc stiffness tensors. The class inherits from the 
    Elasticity class. The convention in HomoPy is that the first
    principal direction is orthogonal to the isotropic plane. 

    The transverse-isotropic stiffness matrix has the following 
    form in the normalized Voigt (Mandel) notation

    .. math::
            \underline{\underline{C}} = \begin{pmatrix}
                                        C_{1111} & 2\nu_{12}(\lambda+G_{23}) & 2\nu_{12}(\lambda+G_{23}) & 0 & 0 & 0 \\
                                        & \lambda+2G_{23} & \lambda & 0 & 0 & 0 \\
                                        & & \lambda+2G_{23} & 0 & 0 & 0 \\
                                        & & & 2G_{23} & 0 & 0 \\
                                        & \mathrm{sym} & & & 2G_{12} & 0 \\
                                        & & & & & 2G_{12} \\
                                        \end{pmatrix},

    where

    .. math::
            \begin{array}{lcl}
                \lambda&=&\dfrac{\nu_{12} \nu_{21} + \nu_{23}}
                {( 1-\nu_{23} - 2 \nu_{12}\nu_{21} ) (1+\nu_{23})} E_2
                \\
                C_{1111} &=& \dfrac{1 - \nu_{23}}{1 - \nu_{23} - 2 \nu_{12} \nu_{21}} E_1
            \end{array}
    
    
    The corresponding compliance matrix is

    .. math::
            \underline{\underline{S}} = \begin{pmatrix}
                                        \frac{1}{E_1} & -\frac{\nu_{12}}{E_1} & -\frac{\nu_{12}}{E_1} & 0 & 0 & 0 \\
                                        & \frac{1}{E_2} & -\frac{\nu_{23}}{E_2} & 0 & 0 & 0 \\
                                        & & \frac{1}{E_2} & 0 & 0 & 0 \\
                                        & & & \frac{1}{2G_{23}} & 0 & 0 \\
                                        & \mathrm{sym} & & & \frac{1}{2G_{12}} & 0 \\
                                        & & & & & \frac{1}{2G_{12}} \\
                                        \end{pmatrix}.

    Parameters
    ----------
    E1 : float
        Young's modulus in longitudinal direction.
    E2 : float
        Young's modulus in transverse direction.
    G12 : float
        Shear modulus in the longitudinal-transverse plane.
    G23 : float
        Shear modulus in the transverse-transverse plane.
    nu12 : float
        Poisson's ratio to express strain in transverse direction caused by load in longitudinal direction (dimensionless).

    Attributes
    ----------
    E1 : float
        Young's modulus in longitudinal direction.
    E2 : float
        Young's modulus in transverse direction.
    G12 : float
        Shear modulus in the longitudinal-transverse plane.
    G23 : float
        Shear modulus in the transverse-transverse plane.
    nu12 : float
        Poisson's ratio to express strain in isotropic plane caused by load in longitudinal direction (dimensionless).
    nu23 : float
        Poisson's ratio to express strain in isotropic plane caused by load in transverse direction (dimensionless).

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
        Young's modulus.
    nu : float
        Poisson's ratio (dimensionless).

    Attributes
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio (dimensionless).
    lam : float
        First Lamé constant.
    mu : float
        Second Lamé constant.
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
            First Lamé constant.
        """
        return self.nu / (1 - 2 * self.nu) * 1 / (1 + self.nu) * self.E

    def _get_mu(self):
        """
        Return the second Lamé constant from other material parameters.

        Returns
        -------
        float
            Second Lamé constant.
        """
        return 1 / 2 * 1 / (1 + self.nu) * self.E
