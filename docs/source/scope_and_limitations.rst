Scope and limitations
=====================

| **General advice**
| The homogenization procedures implemented in HomoPy are based on the theory of linear elasticity. As such, rate dependent effects, damage and non-linear effects in general are not taken into account and should not be modelled using HomoPy. 
| The scope of HomoPy is to model multi-inclusion materials with a special emphasis on fiber reinforced materials. In contrast to other implementations, HomoPy is not limited in the number of inclusion types. Nevertheless, HomoPy is limited in inclusion geometries that are implemented. So far, spheres, spheroids (short and long fibers) and needles (endless fibers) can be selected by the user. The authors would like to emphasize that combinations of different geometries are possible without limitations. 
| HomoPy does not imply a limitation in material symmetry of matrix material nor fiber material when using Mori-Tanaka, but so far only Isotropy and Transverse-Isotropy can be selected. In the case of the Shear-Lag modified Halpin-Tsai procedure, matrix and inclusion must be be isotropic.

| Method specific information is listed below:

| **Mori-Tanaka**
| The Mori-Tanaka scheme goes back to Mori and Tanaka (cf. [Mori1973]_) and is a mean-field homogenization scheme based on Eshelby's solution (cf. [Eshelby1957]_). The implementation so far allows UD (needle), spheroidal and circular inclusions. Our algorithm allows to homogenize materials with different types of fibers/inclusions, each possibily having an individual orientation distrubtion. Being a tensorial homogenization scheme, the fiber orientation tensor is directly included in the calculation and the result is an effective stiffness tensor. The authors would like to emphasize that the classic formulation after [Benveniste1987]_ results in an effective stiffness tensor which violates thermodynamic requirements, i.e. which does not contain the major symmetry, for when multi-inclusion materials or non-isotropic inclusions are used, respectively. Further readings on this attribute are given in [Qiu1990]_ and [Weng1990]_. To compensate this, HomoPy offers an algorithm introduced in [Segura2023]_, which always results in symmetric effective stiffnesses.

| **Halpin-Tsai**
| The Halpin-Tsai method is an empirical approach to homogenize two isotropic materials (cf. [Halpin1969]_). Our approach is modified with the Shear-Lag model after Cox (cf. [Cox1952]_), which is also used in [Fu2002]_ and [Fu2019]_. Being a scalar homogenization scheme, it allows to calculate the effective stiffness in the plane which is orthogonal to the isotropic plane within transverse isotropic materials, as it is the case for unidirectional reinforced polymers. The effective stiffness, or Young's modulus, is then a function of the angle to the reinforcing direction. A fiber distrubtion within the plane is recognized by volume averaging of imaginary plies of individual orientations in analogy to the laminate theory.

