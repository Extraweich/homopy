---
title: 'A Python package for homogenization procedures in fiber reinforced polymers'
tags:
-   Python
-   Mechanics
-   Homogenization
-   Fiber Reinforced Polymers
-   Hybrid materials
-   Mori-Tanaka
-   Halpin-Tsai
authors:
-   name: Nicolas Christ
    orcid: 0000-0002-4713-8096
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
-   name: Benedikt M. Scheuring
    orcid: 0000-0003-2971-1431
    affiliation: "1"
-   name: John Montesano
    orcid: 0000-0003-2723-052X
    affiliation: "3"
-   name: Jörg Hohe
    orcid: 0000-0003-3994-4562
    affiliation: "2"
affiliations:
-   name: Karlsruhe Institute of Technology, Karlsruhe, Germany
    index: 1
-   name: Fraunhofer Institute for Mechanics of Materials, Freiburg, Germany
    index: 2
-   name: University of Waterloo, Waterloo, ON, Canada
    index: 3
date: 23 March 2023
bibliography: paper.bib

---

# Summary

The Python package `HomoPy` is a numerical software tool which provides computational methods in the field of continuum mechanics with a specific emphasize on fiber reinforced composites. Experimental research has shown that hybridization effects for multi-inclusion composites exist [@Summerscales1978; @Fu1998a; @Swolfs2014], which raises the demand to have numerical methods at hand to predict the effective properties of such composites. With a multi-inclusion Mori-Tanaka approach and the laminate theory incorporating the shear-lag modified Halpin-Tsai approach, `HomoPy` provides a solution to this demand. \
The key element of `HomoPy` is the calculation and visualization of effective elastic stiffness properties of hybrid materials, i.e. multi-inclusion composites, using homogenization procedures. The current homogenization implementations are the conventional, three-dimensional Mori-Tanaka approach (cf. @Mori1973) in the formulation of @Benveniste1987 with a possible orientation averaging scheme after @Advani1987. A comprehensive study on the effects of the orientation averaging on homogenization procedures can be found in @Bauer2022a. To circumvent effective stiffness tensors, which are not major-symmetric and therefore violate thermodynamical principles, the algorithm in @Segura2023 was implemented and can be activated by a flag parameter to ensure symmetric stiffnesses. Alternatively, the shear-lag modified Halpin-Tsai approach (cf. @Fu2019) for purely planar information, for which the laminate theory is used to calculate effective stiffness properties, is available. The use field of these tools is academic research. An illustration of the implemented procedures is given in \autoref{fig:MTHT}. \
Furthermore, `HomoPy` has the functionality to visualize the effective stiffness properties following @Boehlke2001, which allows for an intuitive comparison of different composites. The resulting effective properties can then be used in consequent numerical simulations.

![Logo of HomoPy. \label{fig:HP}](images/HomoPy.png){ width=40% }

# Statement of need

Current ecological developments call for technical solutions to reduce the CO2 emissions of future innovations. Significant potential for promoting a better eco-balance lies in the development of new material systems, especially in the field of lightweight construction. Fiber-reinforced polymers (FRP) are a promising class of materials in the field of lightweight construction. The fibers used have high specific strength and stiffness properties, so that less material is required to achieve comparable properties compared with conventional materials, e.g. steel. The polymer serves to hold the fibers in place, transfer stresses between them, and increase the toughness properties of the composite. For more literature on FRP, see @Christensen1980 and @Chawla2019.

A general challenge in using FRP in engineering is that predictive simulations rely on robust material models. Since it is a highly inhomogeneous material, the computational cost increases dramatically if all components are modeled directly, i.e., if a full-field simulation is used. To circumvent this, homogenization methods have been developed in the last decades. The goal of a homogenization method is to calculate the material properties of a synthetic homogeneous material, which should then behave effectively like the inhomogeneous material.

![Schematic of implemented homogenization methods, where $\mathbb{C}_i$ is the stiffness tensor of component $i$, $\bar{\mathbb{C}}$ is the effective stiffness tensor, $\phi_i$ is the orientation angle of fiber $i$ and $a_i$ its aspect ratio. Illustration in reference to @Fu1998b. \label{fig:MTHT}](images/Schematic.png)

`HomoPy` was developed to contain two commonly used homogenization methods, that is, the Mori-Tanka (cf. @Mori1973) for 3D stiffness predictions and a shear-lag modified Halpin-Tsai method (cf. @Cox1952 and @Halpin1969) for planar predictions, i.e. laminate predictions, which is based on the formulations in @Fu2019. The goal of `HomoPy` is to provide an open-source implementation of these methods with a particular focus on hybrid FRP modeling. Other modules offering similar functionalities are available, e.g. `fiberpy` and `mechmean` (cf. @mechmean), but to the author's knowledge none of them provide the capability to model hybrid FRPs consisting of different fiber materials and/or geometries.\
A key element in HomoPy is the implementation of the graphical representation of the effective directional stiffnesses according to @Boehlke2001. Comparing different material systems or FRP tape layup orientations is intuitively easier with a graphical representation than comparing up to 21 stiffness components each. The package `mechkit` (cf. @Bauer2022b) also offers this visualization procedure, but is further capable of visualizing the generalized bulk modulus. `fiberoripy` (cf. @fiberoripy) uses a different general algorithm to represent second and forth order tensors, focusing on fiber orientation tensors rather than stiffness tensors. Other visualization tools can be found in the package `continuum-mechanics` (cf. @continuum_mechanics), though the visualization is designed to represent phase speed tensors. To the authors' knowledge, only `HomoPy` offers a visualization tool for elastic tensors in laminates and 2D stiffness tensors in general, respectively.

To this point, `HomoPy` is limited to calculating the effective elastic properties with the two methods mentioned above. Possible extensions for the future include thermal expansion properties and other homogenization methods.

# Acknowledgements

The research documented in this manuscript has been funded by the German Research Foundation (DFG) within the International Research Training Group “Integrated engineering of continuous-discontinuous long fiber-reinforced polymer structures” (GRK 2078/2). The support by the German Research Foundation (DFG) is gratefully acknowledged.

In addition, N.C. acknowledges Julian Karl Bauer for valuable technical discussions.

# References
