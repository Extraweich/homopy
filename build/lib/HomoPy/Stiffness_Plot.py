# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:24:54 2022

@author: nicolas.christ@kit.edu

3D and Polar plot of Young's modulus body based on Böhlke, Brüggemann (2001).
"""

import numpy as np
from .tensor import Tensor
from .methods import Laminate

sin = np.sin
cos = np.cos


class ElasticPlot(Tensor):
    def __init__(self, USEVOIGT=False):
        self.USEVOIGT = USEVOIGT
        super().__init__()

    def matrix_reduction(self, matrix):
        """
        Return the reduced notation (Voigt or normalized Voigt) depending
        on which one is ought to be used.

        Parameters:
            - matrix : ndarray of shape(3, 3)
                Tensor in regular tensor notation.

        Returns:
            - ... : ndarray of shape(6,19
                Tensor in Voigt or normalized Voigt notation.
        """
        if self.USEVOIGT:
            return self.matrix2voigt(matrix)
        else:
            return self.matrix2mandel(matrix)

    def get_reciprocal_E(self, didi, S):
        """
        Return the reciprocal of Young's modulus (compliance) in
        the direction of di.

        Parameters:
            - didi : ndarray of shape(6, 1)
                Directional tensor.
            - S : ndarray of shape(6, 6)
                Compliance tensor in Voigt or normalized Voigt
                notation.

        Returns:
            - ... : float
                Scalar compliance value in direction of di.
        """
        return np.einsum("i,ij,j->", didi, S, didi)

    def get_E(self, di, S):
        """
        Return Young's modulus in the direction of di.

        Parameters:
            - di : ndarray of shape(3, 1)
                Directional vector.
            - S : ndarray of shape(6, 6)
                Compliance tensor in Voigt or normalized Voigt
                notation.

        Returns:
            - ... : float
                Scalar stiffness value in direction of didi.
        """
        didi = self.matrix_reduction(self.diade(di, di))
        return self.get_reciprocal_E(didi, S) ** (-1)

    def dir_vec(self, phi, theta):
        """
        Return directional vector based on angular parametrization.

        Parameters:
            - phi : float
                First angle in [0, 2*pi].
            - theta : float
                Second angle in [0, pi].

        Returns:
            - ... : ndarray of shape(3,)
                Directional vector.
        """
        return np.array([cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)])

    def plot_E_body(self, S, o, p, bound=[0, 0, 0], rcount=200, ccount=200):
        """
        Plot stiffness body.

        Parameters:
            - S : ndarray of shape(6, 6)
                Compliance tensor in Voigt or normalized Voigt
                notation.
            - o : int
                Number of discretization steps for first angle.
            - p : int
                Number of discretization steps for second angle.
            - bound : array of shape(3,), default=[0,0,0]
                Boundaries for the 3 axis for the visualization.
                If [0,0,0], boundaries are set automatically.
            - rcount, ccount : int
                Maximum number of samples used in each direction.
                If the input data is larger, it will be downsampled
                (by slicing) to these numbers of points. Defaults to 200.

        Returns:
            - None
        """
        n = int(o)
        m = int(p)
        E_x = np.zeros((n + 1, m + 1))
        E_y = np.zeros((n + 1, m + 1))
        E_z = np.zeros((n + 1, m + 1))
        for i in range(0, n + 1, 1):
            for j in range(0, m + 1, 1):
                phi = i / n * 2 * np.pi
                theta = j / m * np.pi
                vec = self.dir_vec(phi, theta)
                E = self.get_E(vec, S)
                E_x[i, j] = E * vec[0]
                E_y[i, j] = E * vec[1]
                E_z[i, j] = E * vec[2]

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm

        x = E_x
        y = E_y
        z = E_z

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the surface
        ax.set_xlabel("E11")
        ax.set_ylabel("E22")
        ax.set_zlabel("E33")

        ax.plot_surface(
            x, y, z, cmap=cm.viridis, antialiased=True, rcount=rcount, ccount=ccount
        )

        if not bound[0] == 0:
            ax.set_xlim(-bound[0], bound[0])
        if not bound[1] == 0:
            ax.set_ylim(-bound[1], bound[1])
        if not bound[2] == 0:
            ax.set_zlim(-bound[2], bound[2])

        plt.show()

    def polar_plot_E_body(self, S, o, angle, bound=[0, 0, 0], plot=True):
        """
        Plot slice of stiffness body.

        Parameters:
            - S : ndarray of shape(6, 6)
                Compliance tensor in Voigt or normalized Voigt
                notation.
            - o : int
                Number of discretization steps for first angle.
            - angle : float
                Angle to determine the angular orientation of the slice.
                Does not work yet.
            - bound : array of shape(3,), default=[0,0,0]
                Boundaries for the 3 axis for the visualization.
                If [0,0,0], boundaries are set automatically.
            - plot : boolean
                Determines whether the plot will be displayed. If False,
                the metadata of the plot will be returned instead.

        Returns:
            - rad : ndarray of shape(n+1,)
                Angular positions for polar plot.
            - E : ndarray of shape(n+1,)
                Sitffness at corresponding angle.
        """
        n = int(o)
        E = np.zeros(n + 1)
        rad = np.zeros(n + 1)

        theta = np.pi / 2 + angle  # changing angle does not work yet
        for i in range(0, n + 1, 1):
            phi = i / n * 2 * np.pi
            vec = self.dir_vec(phi, theta)
            E_temp = self.get_E(vec, S)
            E[i] = E_temp
            rad[i] = phi

        if plot == True:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            ax.plot(rad, E)
            # ax.set_rmax(2)
            # ax.set_rticks([0.5*1e10, 1*1e10, 1.5*1e10, 2*1e10])  # Less radial ticks
            # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.grid(True)

            ax.set_title("Young's modulus over angle", va="bottom")
            plt.show()
        else:
            return rad, E

    def polar_plot(self, data):
        """
        Polar plot of multiple Stiffness bodies in one plot. For this
        use the data generated from polar_plot_E_body or polar_plot_laminate
        with plot=False.

        Parameters:
            - data: list
                Data to be plotted with angluar position, stiffness and
                an optional string for the label in the plot.

        Returns:
            - None
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        for datum in data:
            try:
                ax.plot(datum[0], datum[1], label=datum[2])
            except:
                ax.plot(datum[0], datum[1])
        ax.grid(True)
        ax.set_title("Young's modulus over angle", va="bottom")
        ax.legend()
        plt.show()

    def polar_plot_laminate(self, laminate_stiffness, o, limit=None, plot=True):
        """
        Polar plot stiffness body of laminate. This method should be used
        for laminate results for the Halpin-Tsai homogenization.

        Parameters:
        -----------
            - laminate_stiffness : ndarray of shape(3, 3)
                Planar stiffness matrix in Voigt or normalized Voigt
                notation.
            - o : int
                Number of discretization steps for first angle.
            - limit : float
                Limit of radial axis in polar plot.
            - plot : boolean
                Determines whether the plot will be displayed. If False,
                the metadata of the plot will be returned instead.

        Returns:
            - rad : ndarray of shape(n+1,)
                Angular positions for polar plot.
            - E : ndarray of shape(n+1,)
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
        else:
            return rad, E

    def get_E_laminate(self, C, phi):
        """
        Return Young's modulus of lamina as a function of angle omega.

        Parameters
        ----------
        omega : float
            Angle of orientation in radians.

        Returns
        -------
        E : float
            Young's modulus in angle direction
        """
        C_inv = np.linalg.inv(C)

        theta = np.pi / 2
        vec = self.dir_vec(phi, theta)

        didi = self.diade(vec, vec)
        didi_flat = self.matrix_reduction(didi)
        if self.USEVOIGT == False:
            b = 1 / np.sqrt(2)
        else:
            b = 1
        didi_reduced = np.array([didi_flat[0], didi_flat[1], b * didi_flat[5]])

        E = np.einsum("i,ij,j->", didi_reduced, C_inv, didi_reduced) ** (-1)

        return E
