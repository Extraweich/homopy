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

    def dir_vec(self, theta, phi):
        """
        Return directional vector based on angular parametrization.

        Parameters:
            - theta : float
                First angle.
            - phi : float
                Second angle.

        Returns:
            - ... : ndarray of shape(3,)
                Directional vector.
        """
        return np.array([cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)])

    def plot_E_body(self, S, o, p, bound=[0, 0, 0]):
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
                theta = i / n * 2 * np.pi
                phi = j / m * np.pi
                vec = self.dir_vec(theta, phi)
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
            x,
            y,
            z,
            cmap=cm.viridis,
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

        phi = np.pi / 2 + angle  # changing angle does not work yet
        for i in range(0, n + 1, 1):
            theta = i / n * 2 * np.pi
            vec = self.dir_vec(theta, phi)
            E_temp = self.get_E(vec, S)
            E[i] = E_temp
            rad[i] = theta

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
        use the data generated from polar_plot_E_body with plot=False.

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

    def polar_plot_laminate(self, laminate_stiffness, o):
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
        """
        n = int(o)
        E = np.zeros(n + 1)
        rad = np.zeros(n + 1)

        A = laminate_stiffness
        E11 = (A[0, 0] * A[1, 1] - A[0, 1] ** 2) / A[1, 1]
        E22 = (A[0, 0] * A[1, 1] - A[0, 1] ** 2) / A[0, 0]
        G12 = A[2, 2]
        nu12 = A[0, 1] / A[1, 1]

        for i in range(0, n + 1, 1):
            theta = i / n * 2 * np.pi
            E_temp = self.get_E_lamina(E11, E22, G12, nu12, A, theta)
            E[i] = E_temp
            rad[i] = theta

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(rad, E)
        ax.grid(True)

        ax.set_title("Young's modulus over angle", va="bottom")
        plt.show()

    def get_E_lamina(self, E11, E22, G12, nu12, A, theta):
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
        E = 1 / (
            cos(theta) ** 4 / E11
            + sin(theta) ** 4 / E22
            + 1 / 4 * (1 / G12 - 2 * nu12 / E11) * sin(2 * theta) ** 2
        )

        A_rotated = Laminate.rotate_stiffness(A, theta)

        E = (A_rotated[0] * A_rotated[1] - A_rotated[2] ** 2) / A_rotated[1]

        return E
