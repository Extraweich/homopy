<p align="center">
  <img src="https://github.com/Extraweich/homopy/blob/main/docs/source/images/Homopy_Yellow.svg?raw=true", width="400">
</p>

***
[![PyPI version](https://badge.fury.io/py/homopy.svg)](https://badge.fury.io/py/homopy)
[![Documentation status](https://readthedocs.org/projects/homopy/badge/?version=latest)](https://homopy.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ffc0c7b16d154bc18cccc1e857724d86)](https://www.codacy.com/gh/Extraweich/homopy/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Extraweich/homopy&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Extraweich/homopy/main?labpath=%2Fexamples%2FHybrid.ipynb)

*Your solution for stiffness problems*

HomoPy is a Python package to perform calculations of effective stiffness properties in homogenized materials, with an emphasize on fiber reinforced polymers. Furthermore, the package offers visualisation tools for elastic stiffness tensors, so called Young's modulus' bodies. These allow a comparison of angle dependent stiffnesses of different materials.
Currently, HomoPy offers two types of homogenization procedures:
-   Halpin-Tsai with a Shear-Lag modification
-   Mori-Tanaka

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Extraweich/homopy/blob/main/docs/source/images/Schematic_dark.svg?raw=true">
    <img alt="Schematic of implemented homogenization procedures" src="https://github.com/Extraweich/homopy/blob/main/docs/source/images/Schematic_light.svg?raw=true">
  </picture>
</p>

**Figure 1:** Schematic of implemented homogenization methods. Inspired by [[Fu1998]](#Fu1998).

## Halpin-Tsai
The Halpin-Tsai method is an empirical approach to homogenize two isotropic materials (cf. [[Halpin1969]](#Halpin1969)). Our approach is modified with the Shear-Lag model after Cox (cf. [[Cox1952]](#Cox1952)), which is also used in [[Fu2002]](#Fu2002) and [[Lauke2019]](#Lauke2019). Being a scalar homogenization scheme, it allows to calculate the effective stiffness in the plane which is orthogonal to the isotropic plane within transverse isotropic materials, as it is the case for unidirectional reinforced polymers. The effective stiffness, or Young's modulus, is then a function of the angle to the reinforcing direction. A fiber distrubtion within the plane is recognized by volume averaging of imaginary plies of individual orientations in analogy to the laminate theory.

## Mori-Tanaka
The Mori-Tanaka scheme goes back to Mori and Tanaka (cf. [[Mori1973]](#Mori1973)) and is a mean-field homogenization scheme based on Eshelby's solution (cf. [[Eshelby1957]](#Eshelby1957)). The implementation so far only allows spheroidal inclusions, which in fact is an ellispoid with identical minor axes or ellipsoid of revolution, respectively. Our algorithm allows to homogenize materials with different types of fibers, each possibily having an individual fiber distrubtion. Being a tensorial homogenization scheme, the fiber orientation tensor is directly included in the calculation and the result is an effective stiffness tensor. The authors would like to emphasize that for multi-inclusion problems or non-isotropic inclusions, the effective stiffness tensor may violate thermodynamic requirements, such as a symmetric stiffness tensor. Further readings of this attribute are given in [[Qiu1990]](#Qiu1990) and [[Weng1990]](#Weng1990). To compensate this, HomoPy offers an algorithm introduced in [[Segura2023]](#Segura2023), which always results in symmetric effective stiffnesses.

### Documentation

The documentation can be found in [the docs](https://homopy.readthedocs.io/en/latest/index.html).

### Interactive example

An interactive example to intuitively see the effects of fiber distributions on the effective properties of hybrid materials can be found in [Binder](https://mybinder.org/v2/gh/Extraweich/homopy/main?labpath=%2Fexamples%2FHybrid.ipynb).


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Extraweich/homopy/blob/main/docs/source/images/MTvsHT_dark.svg?raw=true"">
    <img alt="Hybrid polar plot" src="https://github.com/Extraweich/homopy/blob/main/docs/source/images/MTvsHT_light.svg?raw=true"">
  </picture>
</p>


***
Further topic related methods:
-   Closures to calculate orientation tensors of forth order from an orientation tensor of second order are available in [fiberoripy](https://github.com/nilsmeyerkit/fiberoripy)
-   Further tensor operations and output formats are available in [mechkit](https://github.com/JulianKarlBauer/mechkit)

***
<a id="Halpin1969">[Halpin1969]</a>  John C. Halpin, *Effects of environmental factors on composite materials*, 1969. \
<a id="Cox1952">[Cox1952]</a> H. L. Cox, *The elasticity and strength of paper and other fibrous materials*, British Journal of Applied Physics 3 (3) (1952) 72–79. doi:10.1088/05083443/3/3/302. \
<a id="Fu1998">[Fu1998]</a> S.-Y. Fu, B. Lauke, *The elastic modulus of misaligned short-fiber-reinforced polymers*, Composites Science and Technology 58 (3) (1998) 389-40. doi:10.1016/S0266-3538(97)00129-2. \
<a id="Fu2002">[Fu2002]</a> S.-Y. Fu, G. Xu, Y.-W. Mai, *On the elastic modulus of hybrid particle/short-fiber/polymer composites*, Composites Part B: Engineering 33 (4) (2002) 291–299. doi:10.1016/S1359-8368(02)00013-6. \
<a id="Lauke2019">[Lauke2019]</a> S.-Y. Fu, B. Lauke, Y.-W. Mai, *Science and engineering of short fibre-reinforced polymer composites*, Woodhead Publishing (2019). \
<a id="Mori1973">[Mori1973]</a> T. Mori, K. Tanaka, *Average stress in matrix and average elastic energy of materials with misfitting inclusions*, Acta Metallurgica 21 (5) (1973), 571-574. \
<a id="Eshelby1957">[Eshelby1957]</a> J.-D. Eshelby, *The determination of the elastic field of an ellipsoidal inclusion, and related problems*, Proceedings of the Royal Society of London A 241 (1957), 376–396. \
<a id="Qiu1990">[Qui1990]</a> Y. P. Qiu, G. J. Weng, *On the application of mori-tanaka’s theory involving transversely isotropic spheroidal inclusions*, International Journal of Engineering Science 28 (11) (1990) 1121-1137. doi:10.1016/00207225(90)90112-V. \
<a id="Weng1990">[Weng1990]</a> G. J. Weng, *The theoretical connection between mori-tanaka’s theory and the hashin-shtrikman-walpole bounds*, International Journal of Engineering Science 28 (11) (1990) 1111–1120. doi:10.1016/00207225(90)90111-U \
<a id="Segura2023">[Segura2023]</a> N. J. Segura, B. L.A. Pichler and C. Hellmich, *Concentration tensors preserving elastic symmetry of multiphase composites*, Mechanics of Materials 178 (2023), https://doi.org/10.1016/j.mechmat.2023.104555
