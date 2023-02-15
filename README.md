<p align="center">
  <img src="https://github.com/Extraweich/homopy/blob/main/docs/source/images/Homopy_Yellow.svg?raw=true", width="400">
</p>

***
[![PyPI version](https://badge.fury.io/py/homopy.svg)](https://badge.fury.io/py/homopy)
[![Documentation status](https://readthedocs.org/projects/homopy/badge/?version=latest)](https://homopy.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Extraweich/homopy/main?labpath=%2Fexamples%2FHybrid.ipynb)

*Your solution for stiffness problems*

HomoPy is a Python package to perform calculations of effective stiffness properties in homogenized materials, with an emphasize on fiber reinforced polymers. Furthermore, the package offers visualisation tools for elastic stiffness tensors, so called Young's modulus' bodies. These allow a comparison of angle dependent stiffnesses of different materials.
Currently, HomoPy offers two types of homogenization procedures:
- Halpin-Tsai with a Shear-Lag modification
- Mori-Tanaka

![example_image](https://github.com/Extraweich/homopy/blob/main/docs/source/images/MTvsHT.png?raw=true)

## Halpin-Tsai
The Halpin-Tsai method is an empirical approach to homogenize two isotropic materials (cf. [[1]](#1)). Our approach is modified with the Shear-Lag model after Cox (cf. [[2]](#2)), which is also used in [[3]](#3) and [[4]](#4). Being a scalar homogenization scheme, it allows to calculate the effective stiffness in the plane which is orthogonal to the isotropic plane within transverse isotropic materials, as it is the case for unidirectional reinforced polymers. The effective stiffness, or Young's modulus, is then a function of the angle to the reinforcing direction. A fiber distrubtion within the plane is recognized by volume averaging of imaginary plies of individual orientations in analogy to the laminate theory.

## Mori-Tanaka
The Mori-Tanaka scheme goes back to Mori and Tanaka (cf. [[5]](#5)) and is a mean-field homogenization scheme based on Eshelby's solution (cf. [[6]](#6)). The implementation so far only allows spheroidal inclusions, which in fact is an ellispoid with identical minor axes or ellipsoid of revolution, respectively. Our algorithm allows to homogenize materials with different types of fibers, each possibily having an individual fiber distrubtion. Being a tensorial homogenization scheme, the fiber orientation tensor is directly included in the calculation and the result is an effective stiffness tensor. The authors would like to emphasize that for multi-inclusion problems or non-isotropic inclusions, the effective stiffness tensor may violate thermodynamic requirements, such as a symmetric stiffness tensor. Further readings of this attribute are given in [[7]](#7) and [[8]](#8). To compensate this, HomoPy offers an algorithm introduced in [[9]](#9), which always results in symmetric effective stiffnesses.

### Documentation

The documentation can be found in [the docs](https://homopy.readthedocs.io/en/latest/index.html).

### Interactive example

An interactive example to intuitively see the effects of fiber distributions on the effective properties of hybrid materials can be found in [Binder](https://mybinder.org/v2/gh/Extraweich/homopy/main?labpath=%2Fexamples%2FHybrid.ipynb).

***
Further topic related methods:
- Closures to calculate orientation tensors of forth order from an orientation tensor of second order are available in [fiberoripy](https://github.com/nilsmeyerkit/fiberoripy)
- Further tensor operations and output formats are available in [mechkit](https://github.com/JulianKarlBauer/mechkit)

***
<a id="1">[1]</a>  John C. Halpin, *Effects of environmental factors on composite materials*, 1969. \
<a id="2">[2]</a> H. L. Cox, *The elasticity and strength of paper and other fibrous materials*, British Journal of Applied Physics 3 (3) (1952) 72–79. doi:10.1088/05083443/3/3/302. \
<a id="3">[3]</a> S.-Y. Fu, G. Xu, Y.-W. Mai, *On the elastic modulus of hybrid particle/short-fiber/polymer composites*, Composites Part B: Engineering 33 (4) (2002) 291–299. doi:10.1016/S1359-8368(02)00013-6. \
<a id="4">[4]</a> S.-Y. Fu, B. Lauke, Y.-W. Mai, *Science and engineering of short fibre-reinforced polymer composites*, Woodhead Publishing (2019). \
<a id="5">[5]</a> T. Mori, K. Tanaka, *Average stress in matrix and average elastic energy of materials with misfitting inclusions*, Acta Metallurgica 21 (5) (1973), 571-574. \
<a id="6">[6]</a> J.-D. Eshelby, *The determination of the elastic field of an ellipsoidal inclusion, and related problems*, Proceedings of the Royal Society of London A 241 (1957), 376–396. \
<a id="7">[7]</a> Y. P. Qiu, G. J. Weng, *On the application of mori-tanaka’s theory involving transversely isotropic spheroidal inclusions*, International Journal of Engineering Science 28 (11) (1990) 1121-1137. doi:10.1016/00207225(90)90112-V. \
<a id="8">[8]</a> G. J. Weng, *The theoretical connection between mori-tanaka’s theory and the hashin-shtrikman-walpole bounds*, International Journal of Engineering Science 28 (11) (1990) 1111–1120. doi:10.1016/00207225(90)90111-U \
<a id="9">[9]</a> N. J. Segura, B. L.A. Pichler and C. Hellmich, *Concentration tensors preserving elastic symmetry of multiphase composites*, Mechanics of Materials 178 (2023), https://doi.org/10.1016/j.mechmat.2023.104555
