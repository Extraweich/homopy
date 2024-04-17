# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:09:24 2022

@author: nicolas.christ@kit.edu

Tensor class for basic arithmetic operations. More information on tensor representation in Voigt and Mandel notations are given in [Brannon2018]_.
"""

import numpy as np


class Tensor:
    """
    Tensor class to help with basic arithmetic operations on tensor space.

    Attributes
    ----------
    e1 : ndarray of shape (3,)
        Vector 1 of orthonormalbasis of 1st order tensors.
    e2 : ndarray of shape (3,)
        Vector 2 of orthonormalbasis of 1st order tensors.
    e3 : ndarray of shape (3,)
        Vector 3 of orthonormalbasis of 1st order tensors.
    B : ndarray of shape (3, 3, 6)
        Orthonormalbasis of 4th order tensors in Mandel notation.
    B_voigt : ndarray of shape (3, 3, 6)
        Orthogonalbasis of 4th order tensors in Voigt notation.
    """

    def __init__(self):
        self.e1 = np.array([1, 0, 0])
        self.e2 = np.array([0, 1, 0])
        self.e3 = np.array([0, 0, 1])

        # Orthonormalbasis 4th order tensor (Mandel)
        self.B = np.zeros((3, 3, 6))
        self.B[:, :, 0] = self._diade(self.e1, self.e1)
        self.B[:, :, 1] = self._diade(self.e2, self.e2)
        self.B[:, :, 2] = self._diade(self.e3, self.e3)
        self.B[:, :, 3] = (
            np.sqrt(2)
            / 2
            * (self._diade(self.e2, self.e3) + self._diade(self.e3, self.e2))
        )
        self.B[:, :, 4] = (
            np.sqrt(2)
            / 2
            * (self._diade(self.e1, self.e3) + self._diade(self.e3, self.e1))
        )
        self.B[:, :, 5] = (
            np.sqrt(2)
            / 2
            * (self._diade(self.e1, self.e2) + self._diade(self.e2, self.e1))
        )

        # Orthogonalbasis 4th order tensor (Voigt)
        self.B_voigt = np.zeros((3, 3, 6))
        self.B_voigt[:, :, 0] = self._diade(self.e1, self.e1)
        self.B_voigt[:, :, 1] = self._diade(self.e2, self.e2)
        self.B_voigt[:, :, 2] = self._diade(self.e3, self.e3)
        self.B_voigt[:, :, 3] = (self._diade(self.e2, self.e3) + self._diade(self.e3, self.e2))
        self.B_voigt[:, :, 4] = (self._diade(self.e1, self.e3) + self._diade(self.e3, self.e1))
        self.B_voigt[:, :, 5] = (self._diade(self.e1, self.e2) + self._diade(self.e2, self.e1))


    def _diade(self, di, dj):
        """
        Return diadic product of two directional vectors. This is used to
        calculate the basis tensors in the Mandel notation.

        Parameters
        ----------
        di :  ndarray of shape (3,)
            Directional vector #1.
        dj :  ndarray of shape (3,)
            Directional vector #2.


        Returns
        -------
        ndarray of shape (3, 3)
            Tensor of 2nd order in tensor notation.
        """
        return np.einsum("i,j->ij", di, dj)

    def _diade4(self, bi, bj):
        """
        Return diadic product of two tensors. This is used to transfer
        stiffness tensors from Mandel notation to regular tensor
        notation.

        Parameters
        ----------
        bi : ndarray of shape (3, 3)
            Orthonormal basis tensor #1.
        bj : ndarray of shape (3, 3)
            Orthonormal basis tensor #2.

        Returns
        -------
        ndarray of shape (3, 3, 3, 3)
            Tensor of 4th order in tensor notation.
        """
        return np.einsum("ij,kl->ijkl", bi, bj)

    def tensor_product(self, tensor_a, tensor_b):
        """
        Return the mapping of one tensor of 4th order to another in the
        Mandel notation.

        Parameters
        ----------
        tensor_a : ndarray of shape (6, 6)
            Tensor #1.
        tensor_b : ndarray of shape (6, 6)
            Tensor #2.

        Returns
        -------
        ndarray of shape (6, 6)
            Resulting mapping.
        """
        return np.einsum("ij,jk->ik", tensor_a, tensor_b)

    def matrix2voigt(self, matrix):
        """
        Return the Voigt notation of a tensor of 2nd order
        calculated from the regular tensor notation.

        Parameters
        ----------
        matrix : ndarray of shape (3, 3)
            Tensor of 2nd order in regular tensor notation.

        Returns
        -------
        ndarray of shape (6,)
            Tensor in Voigt notation.
        """
        return np.array(
            [
                matrix[0, 0],
                matrix[1, 1],
                matrix[2, 2],
                matrix[1, 2],
                matrix[0, 2],
                matrix[0, 1],
            ]
        )

    def matrix2mandel(self, matrix):
        """
        Return the Mandel notation of a tensor of 2nd order
        calculated from the regular tensor notation.

        Parameters
        ----------
        matrix : ndarray of shape (3, 3)
            Tensor of 2nd order in regular tensor notation.

        Returns
        -------
        ndarray of shape (6,)
            Tensor in Mandel notation.
        """
        b = np.sqrt(2)
        return np.array(
            [
                matrix[0, 0],
                matrix[1, 1],
                matrix[2, 2],
                b * matrix[1, 2],
                b * matrix[0, 2],
                b * matrix[0, 1],
            ]
        )

    def _tensor2matrix(self, tensor, representation):
        """
        Return a matrix representation (either Mandel or Voigt) of a tensor
        of 4th order calculated from the regular tensor notation.

        Parameters
        ----------
        tensor : ndarray of shape (3, 3, 3, 3)
            Tensor of 4th order in regular tensor notation.
        representation : string
            Reduction type (options: 'voigt', 'mandel')

        Returns
        -------
        ndarray of shape (6, 6)
            Tensor in reduced notation.
        """
        assert (
            representation == "voigt" or representation == "mandel"
        ), "Only Mandel or Voigt notation are valid!"

        if representation == "voigt":
            c = 1
            b = 1
        else:
            c = np.sqrt(2)
            b = np.sqrt(2)

        g = tensor

        return np.array(
            [
                [
                    g[0, 0, 0, 0],
                    g[0, 0, 1, 1],
                    g[0, 0, 2, 2],
                    b * g[0, 0, 1, 2],
                    b * g[0, 0, 0, 2],
                    b * g[0, 0, 0, 1],
                ],
                [
                    g[1, 1, 0, 0],
                    g[1, 1, 1, 1],
                    g[1, 1, 2, 2],
                    b * g[1, 1, 1, 2],
                    b * g[1, 1, 0, 2],
                    b * g[1, 1, 0, 1],
                ],
                [
                    g[2, 2, 0, 0],
                    g[2, 2, 1, 1],
                    g[2, 2, 2, 2],
                    b * g[2, 2, 1, 2],
                    b * g[2, 2, 0, 2],
                    b * g[2, 2, 0, 1],
                ],
                [
                    c * g[1, 2, 0, 0],
                    c * g[1, 2, 1, 1],
                    c * g[1, 2, 2, 2],
                    b * c * g[1, 2, 1, 2],
                    b * c * g[1, 2, 0, 2],
                    b * c * g[1, 2, 0, 1],
                ],
                [
                    c * g[0, 2, 0, 0],
                    c * g[0, 2, 1, 1],
                    c * g[0, 2, 2, 2],
                    b * c * g[0, 2, 1, 2],
                    b * c * g[0, 2, 0, 2],
                    b * c * g[0, 2, 0, 1],
                ],
                [
                    c * g[0, 1, 0, 0],
                    c * g[0, 1, 1, 1],
                    c * g[0, 1, 2, 2],
                    b * c * g[0, 1, 1, 2],
                    b * c * g[0, 1, 0, 2],
                    b * c * g[0, 1, 0, 1],
                ],
            ]
        )

    def tensor2mandel(self, tensor):
        """
        Return the Mandel notation of a tensor of 4th
        order calculated from the regular tensor notation.

        Parameters
        ----------
        tensor : ndarray of shape (3, 3, 3, 3)
            Tensor of 4th order in regular tensor notation.

        Returns
        -------
        ndarray of shape (6, 6)
            Tensor in Mandel notation.
        """
        return self._tensor2matrix(tensor, representation="mandel")

    def tensor2voigt(self, tensor):
        """
        Return the Voigt notation of a tensor of 4th
        order calculated from the regular tensor notation.

        Parameters
        ----------
        tensor : ndarray of shape (3, 3, 3, 3)
            Tensor of 4th order in regular tensor notation.

        Returns
        -------
        ndarray of shape (6, 6)
            Tensor in Mandel notation.
        """
        return self._tensor2matrix(tensor, representation="voigt")

    def mandel2tensor(self, mandel):
        """
        Return the regular tensor notation of a tensor calculated from
        the Mandel notation.

        Parameters
        ----------
        mandel : ndarray of shape (6, 6)
            Tensor of 4th order in Mandel notation.

        Returns
        -------
        tensor : ndarray of shape (3, 3, 3, 3)
            Tensor in regular tensor notation.
        """
        tensor = np.zeros((3, 3, 3, 3))
        for i in range(0, 6):
            for j in range(0, 6):
                tensor += mandel[i, j] * self._diade4(self.B[:, :, i], self.B[:, :, j])
        return tensor
    
    def voigt2tensor(self, voigt):
        """
        Return the regular tensor notation of a tensor calculated from
        the Voigt notation.

        Parameters
        ----------
        voigt : ndarray of shape (6, 6)
            Tensor of 4th order in Voigt notation.

        Returns
        -------
        tensor : ndarray of shape (3, 3, 3, 3)
            Tensor in regular tensor notation.
        """
        tensor = np.zeros((3, 3, 3, 3))
        for i in range(0, 6):
            for j in range(0, 6):
                tensor += voigt[i, j] * self._diade4(self.B_voigt[:, :, i], self.B_voigt[:, :, j])
        return tensor

    def mandel2voigt(self, mandel):
        """
        Return the Voigt notation of a matrix based on
        the Mandel notation.

        Parameters
        ----------
        mandel : ndarray of shape (6, 6)
            Tensor of 4th order in Mandel notation.

        Returns
        -------
        voigt : ndarray of shape (6, 6)
            Tensor of 4th order in Voigt notation.
        """
        voigt = self.tensor2voigt(self.mandel2tensor(mandel))
        return voigt

    def voigt2mandel(self, voigt):
        """
        Return the Mandel notation of a matrix based on
        the Voigt notation.

        Parameters
        ----------
        voigt : ndarray of shape (6, 6)
            Tensor of 4th order in Voigt notation.

        Returns
        -------
        mandel : ndarray of shape (6, 6)
            Tensor of 4th order in Mandel notation.
        """
        mandel = self.tensor2mandel(self.voigt2tensor(voigt))
        return mandel
