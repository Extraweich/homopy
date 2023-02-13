# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:09:24 2022

@author: nicolas.christ@kit.edu

Tensor class for basic arithmetic operations. More information on tensor representation in Voigt and Mandel notations are given in [Brannon2018]_.

.. [Brannon2018] Brannon, R. M. (2018), 'Voigt and Mandel components', in *Rotation, Reflection, and Frame Changes*, IOP Publishing
"""

import numpy as np


class Tensor:
    """
    Tensor class to help with basic arithmetic operations on tensor space.
    """

    def __init__(self):
        """
        Initialize the object.

        Parameters:
            - None

        Object variables:
            - e1, e2, e3 : *ndarray of shape (3,)*
                Orthonormalbasis of 1st order tensors (vectors)
            - B : *ndarray of shape (3, 3, 6)*
                Orthonormalbasis of 4th order tensors in normalized Voigt
                notation.

        Returns:
            - None
        """
        self.e1 = np.array([1, 0, 0])
        self.e2 = np.array([0, 1, 0])
        self.e3 = np.array([0, 0, 1])

        # Orthonormalbasis 4th order tensor
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

    def _diade(self, di, dj):
        """
        Return diadic product of two directional vectors. This is used to
        calculate the basis tensors in the normalized Voigt notation.

        :param di: Directional vector #1
        :type di: *ndarray of shape (3,)*
        :param dj: Directional vector #2
        :type dj: *ndarray of shape (3,)*

        Parameters:
            - di : *ndarray of shape (3,)*
                Directional vector #1.
            - dj : *ndarray of shape (3,)*
                Directional vector #2.

        Returns:
            - ... : *ndarray of shape (3, 3)*
                Tensor of 2nd order in tensor notation.
        """
        return np.einsum("i,j->ij", di, dj)

    def _diade4(self, bi, bj):
        """
        Return diadic product of two tensors. This is used to transfer
        stiffness tensors from normalized Voigt notation to regular tensor
        notation.

        Parameters:
            - bi : *ndarray of shape (3, 3)*
                Orthonormal basis tensor #1.
            - bj : *ndarray of shape (3, 3)*
                Orthonormal basis tensor #2.

        Returns:
            - ... : *ndarray of shape (3, 3, 3, 3)*
                Tensor of 4th order in tensor notation.
        """
        return np.einsum("ij,kl->ijkl", bi, bj)

    def tensor_product(self, tensor_a, tensor_b):
        """
        Return the mapping of one tensor of 4th order to another in the
        normalized Voigt notation.

        :param tensor_a: Tensor #1
        :type tensor_a: *ndarray of shape (6, 6)*
        :param tensor_b: Tensor #2
        :type tensor_b: *ndarray of shape (6, 6)*
        :return: Resulting mapping
        :rtype: ndarray of shape (6,6)*
        """
        return np.einsum("ij,jk->ik", tensor_a, tensor_b)

    def matrix2voigt(self, matrix):
        """
        Return the Voigt notation of a tensor of 2nd order
        calculated from the regular tensor notation.

        Parameters:
            - matrix : *ndarray of shape (3, 3)*
                Tensor in regular tensor notation.

        Returns:
            - ... : *ndarray of shape (6,)*
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
        Return the normalized Voigt notation of a tensor of 2nd order
        calculated from the regular tensor notation.

        Parameters:
            - matrix : *ndarray of shape (3, 3)*
                Tensor in regular tensor notation.

        Returns:
            - ... : *ndarray of shape (6,)*
                Tensor in normalized Voigt notation.
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

    def tensor2mandel(self, tensor):
        """
        Return the normalized Voigt (Mandel) notation of a tensor of 4th
        order calculated fromthe regular tensor notation.

        Parameters:
            - tensor : *ndarray of shape (3, 3, 3, 3)*
                Tensor in regular tensor notation.

        Returns:
            - ... : *ndarray of shape (6, 6)*
                Tensor in normalized Voigt notation.
        """
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
                    b * g[1, 2, 0, 0],
                    b * g[1, 2, 1, 1],
                    b * g[1, 2, 2, 2],
                    2 * g[1, 2, 1, 2],
                    2 * g[1, 2, 0, 2],
                    2 * g[1, 2, 0, 1],
                ],
                [
                    b * g[0, 2, 0, 0],
                    b * g[0, 2, 1, 1],
                    b * g[0, 2, 2, 2],
                    2 * g[0, 2, 1, 2],
                    2 * g[0, 2, 0, 2],
                    2 * g[0, 2, 0, 1],
                ],
                [
                    b * g[0, 1, 0, 0],
                    b * g[0, 1, 1, 1],
                    b * g[0, 1, 2, 2],
                    2 * g[0, 1, 1, 2],
                    2 * g[0, 1, 0, 2],
                    2 * g[0, 1, 0, 1],
                ],
            ]
        )

    def mandel2tensor(self, mandel):
        """
        Return the regular tensor notation of a tensor calculated from
        the normalized Voigt (Mandel) notation.

        Parameters:
            - mandel : *ndarray of shape (6, 6)*
                Tensor in normalized Voigt notation.

        Returns:
            - tensor : *ndarray of shape (3, 3, 3, 3)*
                Tensor in regular tensor notation.
        """
        tensor = np.zeros((3, 3, 3, 3))
        for i in range(0, 6):
            for j in range(0, 6):
                tensor += mandel[i, j] * self._diade4(self.B[:, :, i], self.B[:, :, j])
        return tensor
