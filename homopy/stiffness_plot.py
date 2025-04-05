# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:24:54 2022

@author: nicolas.christ@kit.edu

3D and Polar plot of Young's modulus body based on [Boehlke2001]_.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    def __init__(self, USEVOIGT=False, plot_library="matplotlib"):
        self.USEVOIGT = USEVOIGT
        assert (
            plot_library == "matplotlib" or plot_library == "plotly"
        ), "The variable plot_library must be 'matplotlib' or 'plotly'!"
        self.plot_library = plot_library
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

    def plot_E_body(self, C, o, p, bound=None, rcount=200, ccount=200, plot=True):
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
        plot : boolean
            Determines whether the plot will be displayed. If False,
            only the metadata of the plot will be returned.
        """
        S = np.linalg.inv(C)

        n = int(o)
        m = int(p)
        E_x = np.zeros((n + 1, m + 1))
        E_y = np.zeros((n + 1, m + 1))
        E_z = np.zeros((n + 1, m + 1))
        dir_vecs = []
        Es = []
        for i in range(0, n + 1, 1):
            for j in range(0, m + 1, 1):
                phi = i / n * 2 * np.pi
                theta = j / m * np.pi
                vec = self._dir_vec(phi, theta)
                E = self._get_E(vec, S)
                E_x[i, j] = E * vec[0]
                E_y[i, j] = E * vec[1]
                E_z[i, j] = E * vec[2]

                Es.append(E)
                dir_vecs.append(vec)

        d = np.sqrt(E_x**2 + E_y**2 + E_z**2)
        d_normalized = d / d.max()

        max_E = d.max()
        if bound is None:
            bound = [max_E, max_E, max_E]

        if plot:

            if self.plot_library == "matplotlib":
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                # Plot the surface
                ax.set_xlabel("E11")
                ax.set_ylabel("E22")
                ax.set_zlabel("E33")

                ax.plot_surface(
                    E_x,
                    E_y,
                    E_z,
                    facecolors=plt.cm.viridis(d_normalized),
                    antialiased=True,
                    rcount=rcount,
                    ccount=ccount,
                )

                ax.set_xlim(-bound[0], bound[0])
                ax.set_ylim(-bound[1], bound[1])
                ax.set_zlim(-bound[2], bound[2])

                plt.show()

                return np.array(dir_vecs), np.array(Es), fig, ax

            if self.plot_library == "plotly":

                trace = go.Surface(
                    x=E_x, y=E_y, z=E_z, surfacecolor=d, colorscale="Viridis"
                )

                fig = go.Figure(
                    data=trace,
                )

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(
                            exponentformat="e",
                            nticks=8,
                            range=[-bound[0], bound[0]],
                            title=dict(text="E11"),
                        ),
                        yaxis=dict(
                            exponentformat="e",
                            nticks=8,
                            range=[-bound[1], bound[1]],
                            title=dict(text="E22"),
                        ),
                        zaxis=dict(
                            exponentformat="e",
                            nticks=8,
                            range=[-bound[2], bound[2]],
                            title=dict(text="E33"),
                        ),
                        aspectmode="cube",
                    ),
                    coloraxis_colorbar=dict(
                        exponentformat="e",
                    ),
                )

                fig.show()

                return np.array(dir_vecs), np.array(Es), fig

        return np.array(dir_vecs), np.array(Es)

    def plot_E_body_cut(
        self,
        C,
        o,
        p,
        normal=np.array([0, 0, 1]),
        bound=None,
        rcount=200,
        ccount=200,
        remove="positive",
    ):
        """
        Plot stiffness body with a cutting plane.

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

        # rotate from x-y-plane to normal plane
        normal = normal / np.linalg.norm(normal)
        z_vec = np.array([0, 0, 1])
        vec_v = np.cross(z_vec, normal)
        cosine = z_vec.dot(normal)
        v_mat = np.array(
            [
                [0, -vec_v[2], vec_v[1]],
                [vec_v[2], 0, -vec_v[0]],
                [-vec_v[1], vec_v[0], 0],
            ]
        )
        R = (
            np.eye(3) + v_mat + np.dot(v_mat, v_mat) * 1 / (1 + cosine)
        )  # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

        for i in range(0, n + 1, 1):
            for j in range(0, m + 1, 1):
                phi = i / n * 2 * np.pi
                if remove == "positive":
                    theta = j / m * np.pi / 2
                elif remove == "negative":
                    theta = np.pi - j / m * np.pi / 2
                vec = self._dir_vec(phi, theta)
                vec_rot = R @ vec

                E = self._get_E(vec_rot, S)
                E_x[i, j] = E * vec_rot[0]
                E_y[i, j] = E * vec_rot[1]
                E_z[i, j] = E * vec_rot[2]

        d = np.sqrt(E_x**2 + E_y**2 + E_z**2)
        d_normalized = d / d.max()

        max_E = d.max()
        if bound is None:
            bound = [max_E, max_E, max_E]

        x_polar = []
        y_polar = []
        z_polar = []
        theta = np.pi / 2
        for i in range(0, n + 1, 1):
            phi = i / n * 2 * np.pi
            vec = self._dir_vec(phi, theta)
            vec_rot = R @ vec
            E_temp = self._get_E(vec_rot, S)
            vec_rot *= E_temp

            x_polar.append(vec_rot[0])
            y_polar.append(vec_rot[1])
            z_polar.append(vec_rot[2])

        pts = 100
        x_plane = np.outer(np.linspace(-max_E, max_E, pts), np.ones(pts))
        y_plane = x_plane.copy().T
        z_plane = -(x_plane * normal[0] + y_plane * normal[1]) / normal[2]

        if self.plot_library == "matplotlib":

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Plot the surface
            ax.set_xlabel("E11")
            ax.set_ylabel("E22")
            ax.set_zlabel("E33")

            ax.plot_surface(
                E_x,
                E_y,
                E_z,
                facecolors=plt.cm.viridis(d_normalized),
                antialiased=True,
                rcount=rcount,
                ccount=ccount,
            )

            ax.plot_surface(
                x_plane,
                y_plane,
                z_plane,
                color="pink",
                alpha=0.3,
                antialiased=True,
                rcount=rcount,
                ccount=ccount,
            )

            ax.plot(x_polar, y_polar, z_polar, color="red", linewidth=3)

            ax.set_xlim(-bound[0], bound[0])
            ax.set_ylim(-bound[1], bound[1])
            ax.set_zlim(-bound[2], bound[2])

        elif self.plot_library == "plotly":

            trace = go.Surface(
                x=E_x, y=E_y, z=E_z, surfacecolor=d, colorscale="Viridis"
            )

            fig = go.Figure(
                data=trace,
            )

            surfacecolor = np.zeros(z_plane.shape)

            fig.add_trace(
                go.Surface(
                    x=x_plane,
                    y=y_plane,
                    z=z_plane,
                    surfacecolor=surfacecolor,
                    opacity=0.5,
                    showscale=False,
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=x_polar,
                    y=y_polar,
                    z=z_polar,
                    mode="lines",
                    line=dict(width=3),
                )
            )

            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        exponentformat="e",
                        nticks=8,
                        range=[-bound[0], bound[0]],
                        title=dict(text="E11"),
                    ),
                    yaxis=dict(
                        exponentformat="e",
                        nticks=8,
                        range=[-bound[1], bound[1]],
                        title=dict(text="E22"),
                    ),
                    zaxis=dict(
                        exponentformat="e",
                        nticks=8,
                        range=[-bound[2], bound[2]],
                        title=dict(text="E33"),
                    ),
                    aspectmode="cube",
                ),
                coloraxis_colorbar=dict(
                    exponentformat="e",
                ),
            )

            fig.show()

    def polar_plot_E_body(
        self, C, o, normal=np.array([0, 0, 1]), bound=None, plot=True
    ):
        """
        Plot slice of stiffness body.

        Parameters
        ----------
        C : ndarray of shape (6, 6)
            Stiffness tensor in Voigt or normalized Voigt
            notation.
        o : int
            Number of discretization steps for angle within cutting plane.
        normal : ndarray of shape (3,), default=np.array([0,0,1])
            Normal direction defining cutting plane.
        plot : boolean
            Determines whether the plot will be displayed. If False,
            only the metadata of the plot will be returned.

        Returns
        -------
        rad : ndarray of shape (n+1,)
            Angular positions for polar plot.
        E : ndarray of shape (n+1,)
            Sitffness at corresponding angle.
        fig : figure object
        ax : axis object
        """

        S = np.linalg.inv(C)

        n = int(o)
        E = np.zeros(n + 1)
        rad = np.zeros(n + 1)

        normal = normal / np.linalg.norm(normal)

        z_vec = np.array([0, 0, 1])
        vec_v = np.cross(z_vec, normal)
        cosine = z_vec.dot(normal)
        v_mat = np.array(
            [
                [0, -vec_v[2], vec_v[1]],
                [vec_v[2], 0, -vec_v[0]],
                [-vec_v[1], vec_v[0], 0],
            ]
        )
        R = (
            np.eye(3) + v_mat + np.dot(v_mat, v_mat) * 1 / (1 + cosine)
        )  # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

        theta = np.pi / 2
        for i in range(0, n + 1, 1):
            phi = i / n * 2 * np.pi
            vec = self._dir_vec(phi, theta)
            vec_rot = R @ vec
            E_temp = self._get_E(vec_rot, S)
            E[i] = E_temp
            rad[i] = phi

        if plot == True:

            if self.plot_library == "matplotlib":

                fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
                ax.plot(rad, E)
                ax.grid(True)
                ax.set_rlabel_position(90)
                ax.set_title("Young's modulus over angle", va="bottom")
                plt.show()

                return rad, E, fig, ax

            elif self.plot_library == "plotly":

                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(
                        r=E,
                        theta=rad * 180 / np.pi,
                        mode="lines",
                    )
                )

                fig.update_layout(
                    title=dict(text="Young's modulus over angle"),
                    polar=dict(
                        bgcolor="white",
                        angularaxis=dict(
                            linewidth=1,
                            showline=True,
                            linecolor="black",
                            gridcolor="black",
                        ),
                        radialaxis=dict(
                            side="counterclockwise",
                            showline=True,
                            linewidth=1,
                            linecolor="black",
                            gridcolor="black",
                            exponentformat="e",
                            nticks=5,
                            angle=90,
                            tickangle=90,
                            # gridwidth = 2,
                        ),
                    ),
                )

                fig.show()

                return rad, E, fig

        return rad, E

    def polar_plot(self, data, limit=None):
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
        if self.plot_library == "matplotlib":
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            for datum in data:
                try:
                    ax.plot(datum[0], datum[1], label=datum[2])
                except Exception as ex:
                    ax.plot(datum[0], datum[1])
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)

            if limit is not None:
                ax.set_ylim([0, limit])
            ax.grid(True)
            ax.set_rlabel_position(90)
            ax.set_title("Young's modulus over angle", va="bottom")
            ax.legend()
            plt.show()

            return fig, ax

        elif self.plot_library == "plotly":
            fig = go.Figure()
            for datum in data:
                fig.add_trace(
                    go.Scatterpolar(
                        r=datum[1],
                        theta=datum[0] * 180 / np.pi,
                        mode="lines",
                        name=datum[2],
                    )
                )

            fig.update_layout(
                title=dict(text="Young's modulus over angle"),
                scene=dict(
                    xaxis=dict(
                        exponentformat="e",
                    ),
                ),
                polar=dict(
                    bgcolor="white",
                    angularaxis=dict(
                        linewidth=1,
                        showline=True,
                        linecolor="black",
                        gridcolor="black",
                    ),
                    radialaxis=dict(
                        side="counterclockwise",
                        showline=True,
                        linewidth=1,
                        linecolor="black",
                        gridcolor="black",
                        exponentformat="e",
                        nticks=5,
                        angle=90,
                        tickangle=90,
                        # gridwidth = 2,
                    ),
                ),
            )

            if limit is not None:
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            range=[0, limit],
                        )
                    )
                )

            fig.show()

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

            if self.plot_library == "matplotlib":
                fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
                ax.plot(rad, E)
                if limit is not None:
                    ax.set_ylim([0, limit])
                ax.grid(True)
                ax.set_rlabel_position(90)
                ax.set_title("Young's modulus over angle", va="bottom")
                plt.show()

                return rad, E, fig, ax

            elif self.plot_library == "plotly":

                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(
                        r=E,
                        theta=rad * 180 / np.pi,
                        mode="lines",
                    )
                )

                fig.update_layout(
                    title=dict(text="Young's modulus over angle"),
                    polar=dict(
                        bgcolor="white",
                        angularaxis=dict(
                            linewidth=1,
                            showline=True,
                            linecolor="black",
                            gridcolor="black",
                        ),
                        radialaxis=dict(
                            side="counterclockwise",
                            showline=True,
                            linewidth=1,
                            linecolor="black",
                            gridcolor="black",
                            exponentformat="e",
                            nticks=5,
                            angle=90,
                            tickangle=90,
                            # gridwidth = 2,
                        ),
                    ),
                )

                if limit is not None:
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                range=[0, limit],
                            )
                        )
                    )

                fig.show()

                return rad, E, fig

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
