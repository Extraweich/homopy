# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:24:54 2022

@author: nicolas.christ@kit.edu

3D and Polar plot of Young's modulus body based on [Boehlke2001]_.
"""

import numpy as np
from .tensor import Tensor

sin = np.sin
cos = np.cos


class ElasticPlot(Tensor):
    """
    Plot class to visualize tensorial results.

    Parameters
    ----------
    USEVOIGT : boolean
        Flag which determines the usage of Voigt (USEVOIGT=True),
        or Normalized Voigt / Mandel (USEVOIGT=False).

    Attributes
    ----------
    USEVOIGT : boolean
        Flag which determines the usage of Voigt (USEVOIGT=True),
        or Normalized Voigt / Mandel (USEVOIGT=False).
    """

    def __init__(self, USEVOIGT=False):
        self.USEVOIGT = USEVOIGT
        super().__init__()

    def matrix_reduction(self, matrix):
        """
        Return the reduced notation (Voigt or normalized Voigt) depending
        on which one is ought to be used.

        Parameters
        ----------
        matrix : ndarray of shape (3, 3)
            Tensor in regular tensor notation.

        Returns
        -------
        ndarray of shape (6,)
            Tensor in Voigt or normalized Voigt notation.
        """

        if self.USEVOIGT:
            return self.matrix2voigt(matrix)
        else:
            return self.matrix2mandel(matrix)

    @staticmethod
    def _get_reciprocal_E(didi, S):
        """
        Return the reciprocal of Young's modulus (compliance) in
        the direction of di.

        Parameters
        ----------
        didi : ndarray of shape (6,)
            Directional tensor.
        S : ndarray of shape (6, 6)
            Compliance tensor in Voigt or normalized Voigt
            notation.

        Returns
        -------
        float
            Scalar compliance value in direction of di.
        """

        return np.einsum("i,ij,j->", didi, S, didi)

    def _get_E(self, di, S):
        """
        Return Young's modulus in the direction of di.

        Parameters
        ----------
        di : ndarray of shape (3,)
            Directional vector.
        S : ndarray of shape (6, 6)
            Compliance tensor in Voigt or normalized Voigt
            notation.

        Returns
        -------
        float
            Scalar stiffness value in direction of didi.
        """

        didi = self.matrix_reduction(self._diade(di, di))
        return self._get_reciprocal_E(didi, S) ** (-1)

    @staticmethod
    def _dir_vec(phi, theta):
        """
        Return directional vector based on angular parametrization.

        Parameters
        ----------
        phi : float
            First angle in [0, 2*pi].
        theta : float
            Second angle in [0, pi].

        Returns
        -------
        ndarray of shape (3,)
            Directional vector.
        """

        return np.array([cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)])

    def plot_E_body(self, C, o, p, bound=None, rcount=200, ccount=200):
        """
        Plot stiffness body.

        Parameters
        ----------
        C : ndarray of shape (6, 6)
            Stiffness tensor in Voigt or normalized Voigt
            notation.
        o : int
            Number of discretization steps for first angle.
        p : int
            Number of discretization steps for second angle.
        bound : array of shape (3,), default=None
            Boundaries for the 3 axis for the visualization.
            If None, boundaries are set automatically.
        rcount : int
            Maximum number of samples used in first angle direction.
            If the input data is larger, it will be downsampled
            (by slicing) to these numbers of points. Defaults to 200.
        ccount : int
            Maximum number of samples used in second angle direction.
            If the input data is larger, it will be downsampled
            (by slicing) to these numbers of points. Defaults to 200.
        """
        S = np.linalg.inv(C)

        n = int(o)
        m = int(p)
        E_x = np.zeros((n + 1, m + 1))
        E_y = np.zeros((n + 1, m + 1))
        E_z = np.zeros((n + 1, m + 1))
        for i in range(0, n + 1, 1):
            for j in range(0, m + 1, 1):
                phi = i / n * 2 * np.pi
                theta = j / m * np.pi
                vec = self._dir_vec(phi, theta)
                E = self._get_E(vec, S)
                E_x[i, j] = E * vec[0]
                E_y[i, j] = E * vec[1]
                E_z[i, j] = E * vec[2]

        # from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm

        x = E_x
        y = E_y
        z = E_z

        d = np.sqrt(x**2 + y**2 + z**2)
        d = d / d.max()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the surface
        ax.set_xlabel("E11")
        ax.set_ylabel("E22")
        ax.set_zlabel("E33")

        ax.plot_surface(
            x,
            y,
            z,
            facecolors=plt.cm.viridis(d),
            antialiased=True,
            rcount=rcount,
            ccount=ccount,
        )

        if not bound is None:
            ax.set_xlim(-bound[0], bound[0])
            ax.set_ylim(-bound[1], bound[1])
            ax.set_zlim(-bound[2], bound[2])

        plt.show()

    def polar_plot_E_body(self, C, o, angle, plot=True):
        """
        Plot slice of stiffness body.

        Parameters
        ----------
        C : ndarray of shape (6, 6)
            Stiffness tensor in Voigt or normalized Voigt
            notation.
        o : int
            Number of discretization steps for first angle.
        angle : float
            Angle to determine the angular orientation of the slice.
            Does not work yet.
        plot : boolean
            Determines whether the plot will be displayed. If False,
            only the metadata of the plot will be returned.

        Returns
        -------
        rad : ndarray of shape (n+1,)
            Angular positions for polar plot.
        E : ndarray of shape (n+1,)
            Sitffness at corresponding angle.
        """

        S = np.linalg.inv(C)

        n = int(o)
        E = np.zeros(n + 1)
        rad = np.zeros(n + 1)

        theta = np.pi / 2 + angle  # changing angle does not work yet
        for i in range(0, n + 1, 1):
            phi = i / n * 2 * np.pi
            vec = self._dir_vec(phi, theta)
            E_temp = self._get_E(vec, S)
            E[i] = E_temp
            rad[i] = phi

        if plot == True:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(subplot_kw={"projection": "polar"})
            ax.plot(rad, E)
            # ax.set_rmax(2)
            # ax.set_rticks([0.5*1e10, 1*1e10, 1.5*1e10, 2*1e10])  # Less radial ticks
            # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.grid(True)

            ax.set_title("Young's modulus over angle", va="bottom")
            plt.show()

        return rad, E

    @staticmethod
    def polar_plot(data):
        """
        Polar plot of multiple Stiffness bodies in one plot. For this
        use the data generated from polar_plot_E_body or polar_plot_laminate
        with plot=False.

        Parameters
        ----------
        data: list
            Data to be plotted with angluar position, stiffness and
            an optional string for the label in the plot.
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        for datum in data:
            try:
                ax.plot(datum[0], datum[1], label=datum[2])
            except Exception as ex:
                ax.plot(datum[0], datum[1])
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
        ax.grid(True)
        ax.set_title("Young's modulus over angle", va="bottom")
        ax.legend()
        plt.show()

    def polar_plot_laminate(self, laminate_stiffness, o, limit=None, plot=True):
        """
        Polar plot stiffness body of laminate. This method should be used
        for laminate results for the Halpin-Tsai homogenization.

        Parameters
        ----------
        laminate_stiffness : ndarray of shape (3, 3)
            Planar stiffness matrix in Voigt or normalized Voigt
            notation.
        o : int
            Number of discretization steps for first angle.
        limit : float
            Limit of radial axis in polar plot.
        plot : boolean
            Determines whether the plot will be displayed. If False,
            only the metadata of the plot will be returned.

        Returns
        -------
        rad : ndarray of shape (n+1,)
            Angular positions for polar plot.
        E : ndarray of shape (n+1,)
            Sitffness at corresponding angle.
        """

        n = int(o)
        E = np.zeros(n + 1)
        rad = np.zeros(n + 1)

        C = laminate_stiffness

        for i in range(0, n + 1, 1):
            phi = i / n * 2 * np.pi
            E_temp = self.get_E_laminate(C, phi)
            E[i] = E_temp
            rad[i] = phi

        if plot == True:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            ax.plot(rad, E)
            if limit is not None:
                ax.set_ylim([0, limit])
            ax.grid(True)

            ax.set_title("Young's modulus over angle", va="bottom")
            plt.show()

        return rad, E

    def get_E_laminate(self, C, phi):
        """
        Return Young's modulus of laminate as a function of angle omega.

        Parameters
        ----------
        C : ndarray of shape (3, 3)
            Stiffness of laminate in default (orthonormal) coordinate system.
        phi : float
            Angle of orientation in radians.

        Returns
        -------
        E : float
            Young's modulus in angle direction
        """

        C_inv = np.linalg.inv(C)

        theta = np.pi / 2
        vec = self._dir_vec(phi, theta)

        didi = self._diade(vec, vec)
        didi_flat = self.matrix_reduction(didi)

        didi_reduced = np.array([didi_flat[0], didi_flat[1], didi_flat[5]])

        E = np.einsum("i,ij,j->", didi_reduced, C_inv, didi_reduced) ** (-1)

        return E
