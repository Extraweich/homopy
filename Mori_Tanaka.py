# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:09:24 2022

@author: chri

Mori-Tanaka Homogenization after Hohe (2020) and Seelig (2016).
Eshelby Tensor is taken from Tandon, Weng (1984) but can also be found in Seelig (2016).

Tested:
    - Young's modulus for "almost"sphere (a = 1) in correspondance to Isotropic implementation (Übung MMM)
"""

import numpy as np

class Tensor():
    '''
    Tensor class to help with basic arithmetic operations on tensor space.
    '''
    
    def __init__(self):
        '''
        Initialize the object.
        
        Object variables:
            - e1, e2, e3 : ndarray of shape(3,)
                Orthonormalbasis of 1st order tensors (vectors)
            - B : ndarray of shape(3, 3, 6)
                Orthonormalbasis of 4th order tensors in normalized Voigt
                notation.
                
        Returns:
            - None
        '''
        self.e1 = np.array([1,0,0])
        self.e2 = np.array([0,1,0])
        self.e3 = np.array([0,0,1])
        
        # Orthonormalbasis 4th order tensor
        self.B = np.zeros((3,3,6))
        self.B[:,:,0] = self.diade(self.e1, self.e1)
        self.B[:,:,1] = self.diade(self.e2, self.e2)
        self.B[:,:,2] = self.diade(self.e3, self.e3)
        self.B[:,:,3] = np.sqrt(2)/2*(self.diade(self.e2, self.e3) \
                                      + self.diade(self.e3, self.e2))
        self.B[:,:,4] = np.sqrt(2)/2*(self.diade(self.e1, self.e3) \
                                      + self.diade(self.e3, self.e1))
        self.B[:,:,5] = np.sqrt(2)/2*(self.diade(self.e1, self.e2) \
                                      + self.diade(self.e2, self.e1))
            
    def diade(self, di, dj):
        '''
        Return diadic product of two directional vectors. This is used to
        calculate the basis tensors in the normalized Voigt notation.
        
        Parameters:
            - di : ndarray of shape(3,)
                Directional vector #1.
            - dj : ndarray of shape(3,)
                Directional vector #2.
                
        Returns:
            - ... : ndarray of shape(3, 3)
                Tensor of 2nd order in tensor notation.
        '''
        return np.einsum("i,j->ij", di, dj)

    def diade4(self, bi, bj):
        '''
        Return diadic product of two tensors. This is used to transfer
        stiffness tensors from normalized Voigt notation to regular tensor
        notation.
        
        Parameters:
            - bi : ndarray of shape(3, 3)
                Orthonormal basis tensor #1.
            - bj : ndarray of shape(3, 3)
                Orthonormal basis tensor #2.
                
        Returns:
            - ... : ndarray of shape(3, 3, 3, 3)
                Tensor of 4th order in tensor notation.
        '''
        return np.einsum("ij,kl->ijkl", bi, bj)
    
    def tensorProduct(self, tensorA, tensorB):
        '''
        Return the mapping of one tensor of 2nd order to another in the 
        normalized Voigt notation. 
        
        Parameters:
            - tensorA : ndarray of shape(6, 6)
                Tensor #1.
            - tensorB : ndarray of shape(6, 6)
                Tensor #2
                
        Returns:
            - ... : ndarray of shape(6,6)
                Resulting mapping.
        '''
        return np.einsum('ij,jk->ik',tensorA,tensorB)
        
    def tensor2mandel(self, tensor):
        '''
        Return the normalized Voigt notation of a tensor calculated from
        the regular tensor notation.
        
        Parameters:
            - tensor : ndarray of shape(3, 3, 3, 3)
                Tensor in regular tensor notation.
                
        Returns:
            - ... : ndarray of shape(6, 6)
                Tensor in normalized Voigt notation.
        '''
        b = np.sqrt(2)
        g = tensor
        return np.array([[g[0,0,0,0], g[0,0,1,1], g[0,0,2,2], b*g[0,0,1,2], b*g[0,0,0,2], b*g[0,0,0,1]],
                         [g[1,1,0,0], g[1,1,1,1], g[1,1,2,2], b*g[1,1,1,2], b*g[1,1,0,2], b*g[1,1,0,1]],
                         [g[2,2,0,0], g[2,2,1,1], g[2,2,2,2], b*g[2,2,1,2], b*g[2,2,0,2], b*g[2,2,0,1]],
                         [b*g[1,2,0,0], b*g[1,2,1,1], b*g[1,2,2,2], 2*g[1,2,1,2], 2*g[1,2,0,2], 2*g[1,2,0,1]],
                         [b*g[0,2,0,0], b*g[0,2,1,1], b*g[0,2,2,2], 2*g[0,2,1,2], 2*g[0,2,0,2], 2*g[0,2,0,1]],
                         [b*g[0,1,0,0], b*g[0,1,1,1], b*g[0,1,2,2], 2*g[0,1,1,2], 2*g[0,1,0,2], 2*g[0,1,0,1]]])
    
    def mandel2tensor(self, mandel):
        '''
        Return the regular tensor notation of a tensor calculated from
        the normalized Voigt notation.
        
        Parameters:
            - mandel : ndarray of shape(6, 6)
                Tensor in normalized Voigt notation.
                
        Returns:
            - tensor : ndarray of shape(3, 3, 3, 3)
                Tensor in regular tensor notation.
        '''
        tensor = np.zeros((3,3,3,3))
        for i in range(0, 6):
            for j in range(0, 6):
                tensor += mandel[i,j]*self.diade4(self.B[:,:,i], self.B[:,:,j])
        return tensor
    
    
        

class Elasticity(Tensor):
    '''
    Elasticity class to express generic elasitc stiffness tensors. The class
    inherits from the Tensor class.
    '''
    
    def __init__(self):
        '''
        Initialize the object and call super class initialization.
        
        Object variables:
            - stiffness3333 : ndarray of shape(3, 3, 3, 3)
                Holds the stiffness values in the regular tensor notation.
            - stiffness66 : ndarray of shape(6, 6)
                Holds the stiffness values in the normalized Voigt notation.
                
        Returns:
            - None
        '''
        super().__init__()
        self.stiffness3333 = np.zeros((3,3,3,3))
        self.stiffness66 = np.zeros((6,6))
        
        
class Transverse_Isotropy(Elasticity):
    '''
    Transverse Isotropy class to express transverse-isotropic elasitc stiffness tensors.
    The class inherits from the Elasticity class.
    '''
    
    def __init__(self, E1, E2, G12, G23, nu12):
        '''
        Initialize the object and call super class initialization.
        
        Parameters:
            - E1 : float
                Young's modulus in longitudinal direction.
            - E2 : float
                Young's modulus in transverse direction
            - G12 : float
                Shear modulus in the longitudinal-transverse plane.
            - G23 : float
                Shear modulus in the transverse-transverse plane.
            - nu12 : float
                Poisson's ratio in longitudinal direction.
        
        Object variables:
            - E1 : float
                Young's modulus in longitudinal direction.
            - E2 : float
                Young's modulus in transverse direction
            - G12 : float
                Shear modulus in the longitudinal-transverse plane.
            - G23 : float
                Shear modulus in the transverse-transverse plane.
            - nu12 : float
                Poisson's ratio in longitudinal direction.
            - nu23 : float
                Poisson's ratio in transverse direction
                
        Returns:
            - None
        '''
        super().__init__()
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.G23 = G23
        self.nu12 = nu12
        self.nu21 = self.E2/self.E1*self.nu12
        self.nu23 = self.E2/(2*self.G23) - 1
        self.get_stiffness()
        
    def get_stiffness(self):
        '''
        Calculate the stiffness parameters for both notation.
        
        Parameters:
            - None
            
        Returns:
            - None
        '''
        C1111 = (1-self.nu23)/(1-self.nu23-2*self.nu12*self.nu21)*self.E1
        lam = (self.nu12*self.nu21+self.nu23)/(1-self.nu23-2*self.nu12*self.nu21)\
            /(1+self.nu23)*self.E2
        self.stiffness66 = np.array([[C1111, 2*self.nu12*(lam+self.G23), 2*self.nu12*(lam+self.G23), 0, 0, 0],
                                     [2*self.nu12*(lam+self.G23), lam+2*self.G23, lam, 0, 0, 0],
                                     [2*self.nu12*(lam+self.G23), lam, lam+2*self.G23, 0, 0, 0],
                                     [0, 0, 0, 2*self.G23, 0, 0],
                                     [0, 0, 0, 0, 2*self.G12, 0],
                                     [0, 0, 0, 0, 0, 2*self.G12]])
        self.stiffness3333 = self.mandel2tensor(self.stiffness66)
        

class Isotropy(Transverse_Isotropy):
    '''
    Isotropy class to express isotropic elasitc stiffness tensors.
    The class inherits from the Transverse Isotropy class.
    '''
    
    def __init__(self, E, nu):
        '''
        Initialize the object and call super class initialization.
        
        Parameters:
            - E : float
                Young's modulus.
            - nu : float
                Poisson's ratio.
        
        Object variables:
            - E : float
                Young's modulus.
            - nu : float
                Poisson's ratio.
            - lam : float
                First Lamé constant.
            - mu : float
                Second Lamé constant.
                
        Returns:
            - None
        '''
        self.E = E
        self.nu = nu
        self.lam = self.get_lambda()
        self.mu = self.get_mu()
        super().__init__(self.E, self.E, self.mu, self.mu, self.nu)
        
    def get_lambda(self):
        '''
        Return the first Lamé constant from other material parameters.
        
        Parameters:
            - None
            
        Returns:
            - ... : float
                First Lamé constant.
        '''
        return self.nu/(1-2*self.nu)*1/(1+self.nu)*self.E

    def get_mu(self):
        '''
        Return the second Lamé constant from other material parameters.
        
        Parameters:
            - None
            
        Returns:
            - ... : float
                Second Lamé constant.
        '''
        return 1/2*1/(1+self.nu)*self.E
    
    
    
class Mori_Tanaka(Tensor):
    '''
    Mori Tanaka class to calculate the homogenized stiffness for fiber reinforced 
    polymers with possibly different types of inclusions. The class inherits from
    the Tensor class.
    '''
    
    def __init__(self, matrix, fiber, v_frac, a_ratio):
        '''
        Initialize the object and call super class initialization.
        
        Parameters:
            - matrix : class object of the Elasticity class (or any child class)
                Polymer matrix material.
            - fiber : class object of the Elasticity class (or any child class)
                      or list of objects of the Elasticity class
                Fiber material.
            - v_frac : float
                Volume fraction of the fiber material within the matrix
                material.
            - a_ratio : float
                Aspect ratio of the fiber material.
        
        Object variables:
            - matrix : class object of the Elasticity class (or any child class)
                Polymer matrix material.
            - fiber : class object of the Elasticity class (or any child class)
                Fiber material.
            - Cm : ndarray of shape(6, 6)
                Stiffness of matrix material in normalized Voigt notation.
            - eye : ndarray of shape(6, 6)
                Identity tensor in normalized Voigt notation.
                
        Returns:
            - None
        '''
        super().__init__()
        self.matrix = matrix
        self.fiber = fiber
        self.Cm = matrix.stiffness66
        self.eye = np.eye(6)
        
        # when the fiber parameter is a list, differnt types of inclusions are considered
        if not type(fiber) == list:
            self.fiber = fiber
            self.Cf = fiber.stiffness66
            self.v_frac = v_frac
            self.a_ratio = a_ratio
            self.eshelby66 = self.get_eshelby(self.a_ratio)
            
        else:
            assert len(fiber) == len(v_frac) == len(a_ratio), \
                "Dimensions of stiffnesses, v_fracs and a_ratios do not match!"
            self.nr_constituents = len(fiber)
            self.A_f_alpha = list()
            self.pol_alpha = list()
            self.c_alpha = list() # vol_frac of phase alpha in reference to total inclusion vol_frac
            self.c_f= sum(v_frac)
            
            Cm_inv = np.linalg.inv(self.Cm)
            for i in range(self.nr_constituents):
                Cf_alpha = fiber[i].stiffness66
                pol = Cf_alpha-self.Cm
                self.pol_alpha.append(pol)
                S = self.get_eshelby(a_ratio[i])
                A_inv = self.eye + self.tensorProduct(S, \
                                         self.tensorProduct(Cm_inv, pol))
                A = np.linalg.inv(A_inv)
                self.A_f_alpha.append(A)
                self.c_alpha.append(v_frac[i]/self.c_f)
            
                
    def get_eshelby(self, a_ratio, return_dim='66',shape='ellipsoid'):
        '''
        Return the Eshelby tensor according to the fiber type.
        
        Parameters:
            - a_ratio : float
                Aspect ratio of fiber.
            - return_dim : string, default='66'
                Flag to determine whether the tensor should be returned in
                normalized Voigt or regular tensor notation (options: '66', '3333')
            - shape : string, default='ellipsoid'
                Flag to determine which assumptions are taken into consideration
                for the geometry of the fiber. So far not in use...
                
        Returns:
            - S : ndarray of shape(6, 6) or (3, 3, 3, 3)
                Eshelby inclusion tensor.
                
        '''
        nu = self.matrix.nu
        a = a_ratio
        a2 = a**2
        g = a/(a2-1)**(3/2)*(a*(a2-1)**(1/2)-np.arccosh(a))
        S = np.zeros((3,3,3,3))
        S[0,0,0,0] = 1/(2*(1-nu))*(1-2*nu+(3*a2-1)/(a2-1)-(1-2*nu+3*a2/(a2-1))*g)
        S[1,1,1,1] = 3/(8*(1-nu))*a2/(a2-1)+1/(4*(1-nu))*(1-2*nu-9/(4*(a2-1)))*g
        S[2,2,2,2] = S[1,1,1,1]
        S[1,1,2,2] = 1/(4*(1-nu))*(a2/(2*(a2-1))-(1-2*nu+3/(4*(a2-1)))*g)
        S[2,2,1,1] = S[1,1,2,2]
        S[1,1,0,0] = -1/(2*(1-nu))*a2/(a2-1)+1/(4*(1-nu))*(3*a2/(a2-1)-(1-2*nu))*g
        S[2,2,0,0] = S[1,1,0,0]
        S[0,0,1,1] =-1/(2*(1-nu))*(1-2*nu+1/(a2-1))+1/(2*(1-nu))*(1-2*nu+3/(2*(a2-1)))*g
        S[0,0,2,2] = S[0,0,1,1]
        S[1,2,1,2] = 1/(4*(1-nu))*(a2/(2*(a2-1))+(1-2*nu-3/(4*(a2-1)))*g)
        S[2,1,2,1] = S[1,2,1,2]
        S[0,1,0,1] = 1/(4*(1-nu))*(1-2*nu-(a2+1)/(a2-1)-1/2*(1-2*nu-3*(a2+1)/(a2-1))*g)
        S[0,2,0,2] = S[0,1,0,1]
        if return_dim == '66':
            return self.tensor2mandel(S)
        elif return_dim == '3333':
            return S
    
    def get_effective_stiffness(self):
        '''
        Return the effective stiffness of a inhomogeneous material.
        
        Parameters:
            - None
            
        Returns:
            - C_eff : ndarray of shape (6, 6)
                Homogenized stiffness tensor in the normalized Voigt notation.
        '''
        if not type(self.fiber) == list:
            Cm_inv = np.linalg.inv(self.Cm)
            pol = self.Cf - self.Cm
            A_inv = self.eye + self.tensorProduct(self.eshelby66, \
                                             self.tensorProduct(Cm_inv, pol))
            #A = np.linalg.inv(A_inv)
            C_eff = self.Cm + self.v_frac \
                *self.tensorProduct(pol, np.linalg.inv(self.v_frac\
                                                       *self.eye+(1-self.v_frac)*A_inv))
        else:
            pol_A_ave = np.zeros((6,6))
            A_ave = np.zeros((6,6))
            # calculating the averages
            for i in range(self.nr_constituents):
                A_ave += self.c_alpha[i]*self.A_f_alpha[i]
                pol_A_ave += self.c_alpha[i]\
                    *self.tensorProduct(self.pol_alpha[i], self.A_f_alpha[i])
                
            C_eff = self.Cm + self.tensorProduct(self.c_f*pol_A_ave,
                                                 np.linalg.inv(self.c_f*A_ave\
                                                               +(1-self.c_f)*self.eye))
            
        return C_eff
    
    def get_average_stiffness(self, C_eff, N2, N4):
        '''
        Return the averaged effective stiffness based on orientation tensors.
        
        Parameters:
            - C_eff : ndarray of shape(6, 6) or (3, 3, 3, 3)
                Effective stiffness in normalized Voigt or regular tensor notation.
            - N2 : ndarray of shape(3, 3)
                Orientation tensor of 2nd order.
            - N4 : ndarray of shape(3, 3, 3, 3)
                Orientation tensor of 4th order.
                
        Returns:
            - ... : ndarray of shape (6, 6)
                Averaged stiffness tensor in the normalized Voigt notation.
        '''
        if C_eff.shape == (6,6):
            C_eff = self.mandel2tensor(C_eff)
        
        b1 = C_eff[0,0,0,0] + C_eff[1,1,1,1] - 2*C_eff[0,0,1,1] - 4*C_eff[0,1,0,1]
        b2 = C_eff[0,0,1,1] - C_eff[1,1,2,2]
        b3 = C_eff[0,1,0,1] + 1/2*(C_eff[1,1,2,2] - C_eff[1,1,1,1])
        b4 = C_eff[1,1,2,2]
        b5 = 1/2*(C_eff[1,1,1,1] - C_eff[1,1,2,2])
        
        eye2 = np.eye(3)
        
        C_eff_ave = b1*N4 + b2*(np.einsum('ij,kl->ijkl',N2,eye2) \
                                + np.einsum('ij,kl->ijkl',eye2,N2)) \
            + b3*(np.einsum('ik,lj->ijkl',N2,eye2) \
                  + np.einsum('ik,lj->ijlk',N2,eye2) \
                      + np.einsum('ik,lj->klij',eye2,N2) \
                          + np.einsum('ik,lj->ijlk',eye2,N2)) \
                + b4*(np.einsum('ij,kl->ijkl',eye2,eye2)) \
                      + 2*b5*1/2*(np.einsum('ik,lj->ijkl',eye2,eye2) \
                                  + np.einsum('ik,lj->ijlk',eye2,eye2))
        return self.tensor2mandel(C_eff_ave)
        
    
    
    
if __name__ == "__main__":
    #%%
    Carbon_fiber = Isotropy(242e9, 0.1)
    Glass_fiber = Isotropy(80e9, 0.22)
    Polyamid6 = Isotropy(1.18e9, 0.35)
    
    MT = Mori_Tanaka(Polyamid6, Carbon_fiber, 0.25, 347)
    MT2 = Mori_Tanaka(Polyamid6, Glass_fiber, 0.25, 225)
    C_eff = MT.get_effective_stiffness()
    C2_eff = MT2.get_effective_stiffness()
    S_eff = np.linalg.inv(C_eff)
    S2_eff = np.linalg.inv(C2_eff)
    
    from Stiffness_Plot import plot_E_body, polar_plot_E_body, polar_plot
    from Tsai_Hill import *
    
    TH_Carb = Tsai_Hill(242*1e9, 1.18*1e9, 105*1e9, 0.4*1e9, 0.1, 0.35, 2.5*1e-3,
                    7.2/2*1e-6, 0.25)
    angles = np.arange(0,2*np.pi,0.01)
    Es_Carb = TH_Carb.get_E(angles)
    
    #plot_E_body(S_eff, 200, 100,[6e10,6e10,6e10])   
    pC = polar_plot_E_body(S_eff, 400, 0, plot=False)
    pG = polar_plot_E_body(S2_eff, 400, 0, plot=False)
    polar_plot([pC+('MT Carbon',), pG+('MT Glass',)]) #(angles, Es_Carb, 'Shear-lag')
    
    from fiberpy.mechanics import *
    
    rve_data = {
        "rho0": 1.14e-9,
        "E0": 1.18e9,
        "nu0": 0.35,
        "rho1": 2.55e-9,
        "E1": 242e9,
        "nu1": 0.1,
        "vf": 0.25,
        "aspect_ratio": 347,
    }
    fiber = FiberComposite(rve_data)
        
    
    
    Esh_own = MT.eshelby66
    Esh_pack = MT.tensor2mandel(fiber.Eshelby())
    
    #%% Orientierung
    from fiberoripy.closures import (
        IBOF_closure,
        compute_closure,
        hybrid_closure,
        linear_closure,
        quadratic_closure,
    )
    
    
    N2 = np.eye(3)
    N2[0,0] = 19/32; N2[1,1] = 10/32; N2[2,2] = 3/32
    #N4 = quadratic_closure(N2)
    N4 = IBOF_closure(N2)
    
    ud = fiber.MoriTanaka()
    ud_ave = fiber.ABar(np.array([0.5,0.5,0.]),model="MoriTanaka",closure="invariants")
    ud_inv = np.linalg.inv(ud)
    #plot_E_body(ud_inv, 200, 100,[6e10,6e10,6e10])
    
    C_eff_ave = MT.get_average_stiffness(C_eff, N2, N4)
    S_eff_ave = np.linalg.inv(C_eff_ave)
    S_mat = np.linalg.inv(Polyamid6.stiffness66)
    p2 = polar_plot_E_body(S_eff_ave, 400, 0, plot=False)
    p3 = polar_plot_E_body(S_mat, 400, 0, plot=False)
    polar_plot([pC+('MT UD',), p2+('MT planar iso',), p3+('PA6',)])
    
    #%% Hybrid
    MTH = Mori_Tanaka(Polyamid6, [Glass_fiber, Carbon_fiber], [0.125, 0.125], [225, 347])
    C_eff_H = MTH.get_effective_stiffness()
    S_eff_H = np.linalg.inv(C_eff_H)
    
    pH = polar_plot_E_body(S_eff_H, 400, 0, plot=False)
    polar_plot([pG+('MT Glass',), pC+('MT Carbon',), pH+('MT Hybrid',)])
    

