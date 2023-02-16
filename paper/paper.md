---
title: 'HomoPy: A Python package for homogenization procedures in fiber reinforced polymers'
tags:
-   Python
-   Mechanics
-   Homogenization
-   Fiber Reinforced Polymers
-   Mori-Tanaka
-   Halpin-Tsai
authors:
-   name: Nicolas Christ
    orcid: 0000-0002-4713-8096
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
-   name: Karlsruhe Institute of Technology, Karlsruhe, Germany
    index: 1
-   name: Fraunhofer Institute for Mechanics of Materials, Freiburg, Germany
    index: 2
date: 07 January 2023
bibliography: paper.bib

---

# Summary

The Python package `HomoPy` is a numerical software tool which provides computational methods in the field of continuum mechanics with a specific emphasize on fiber reinforced composites. The key element of `HomoPy` is the calculation and visualization of effective elastic stiffness properties using homogenization procedures. The current homogenization implementations are the conventional, three-dimensional Mori-Tanaka approach with a possible orientation averaging scheme after Advani and Tucker (...) and the shear-lag modified Halpin-Tsai approach for purely planar information, for which the laminate theory is used to calculate effective stiffness properties. The use field of these tools is academic research.

Experimental research has shown that hybridization effects for multi-inclusion composites exist (source), which raises the demand to have numerical methods at hand to predict the effective properties of such composites. With a multi-inclusion Mori-Tanaka approach and the laminate theory incorporating the shear-lag modified Halpin-Tsai approach, `HomoPy` provides a solution to this demand.

Furthermore, `HomoPy` has the functionality to visualize the effective stiffness properties following (Böhlke et al., ...), which allows for an easy comparison of different composites. The resulting effective properties can then used in consequent numerical simulations.

# Statement of need

Current ecological developments call for technical solutions to reduce the ecological footprint of future innovations. Significant opportunities to promote a better eco-balance lie in the development of new material systems, particularly in the area of lightweight construction. Fiber-reinforced polymers (FRP) are a promising class of materials in the field of lightweight construction. The fibers used have high specific strength and stiffness properties, so that less material is required to achieve comparable properties in comparison with conventional materials, e.g. steel. The polymer, hereafter referred to as the matrix, is used to hold the fibers in place and transfer stresses between them. Further literature on FRP can be found in Christensen (cf. 1) and Chawla (cf. 2).

A general challenge in using FRP in engineering is that prediction simulations rely on robust material models. Since it is a highly inhomogeneous material, the computational cost increases dramatically if all components are modeled directly. To circumvent this, homogenization methods have been developed in the last decades. The goal of a homogenization method is to calculate the material properties of a synthetic homogeneous material, which should then effectively behave like the inhomogeneous material.

`HomoPy` was developed to implement two commonly used homogenization methods, namely the Mori-Tanka (cf. 4) for 3D stiffness predictions and a shear-lag modified Halpin-Tsai method (cf. 4 and 5) for planar predictions, i.e. laminate predictions. The goal of `HomoPy` is to provide an open-source implementation of these methods with a particular focus on FRP modeling. Other modules are available, e.g. `fiberpy` and `mechmean`, but to the author's knowledge none of them provides the capability to model hybrid FRPs consisting of different fiber materials and/or geometries. Furthermore, a major advantage in HomoPy is the implementation of the graphical representation of the effective directional stiffnesses according to Böhlke and Brüggemann (cf. 6). Comparing different material systems or FRP tape layup orientations is obviously easier with a graphical representation than comparing up to 21 stiffness components each.

To this point, `HomoPy` is limited to calculating the effective elastic properties with the two methods mentioned above. Possible extensions for the future include thermal expansion properties and other homogenization methods.

# Acknowledgements

The research documented in this manuscript has been funded by the German Research Foundation (DFG) within the International Research Training Group “Integrated engineering of continuous-discontinuous long fiber-reinforced polymer structures” (GRK 2078/2). The support by the German Research Foundation (DFG) is gratefully acknowledged.
