# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:24:30 2022

@author: chri

Modified Tsai-Hill Criterion after Yan, Yang et al. (2018)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

sin = np.sin
cos = np.cos
tanh = np.tanh

class Tsai_Hill():
    
    def __init__(self, E_f, E_m, G_f, G_m, nu_f, nu_m, l_f, r_f, vol_f,
                 package='hex'):
        '''
        Class to perform the Tsai_Hill homogenization.
        
        Parameters
        ----------
        E_f : Float
            Young's modulus of fiber.
        E_m : Float
            Young's modulus of matrix.
        G_f : Float
            Shear modulus of fiber.
        G_m : Float
            Shear modulus of matrix.
        nu_f : Float
            Poisson ratio of fiber.
        nu_m : Float
            Poisson ratio of matrix.
        l_f : Float
            Average length of fiber.
        r_f : Float
            Average radius of fiber.
        vol_f : Float
            Poisson ratio of matrix.
        package : String (default: hex), other options: square
            Package structure of fibers in composite.
        '''
        self.E_f = E_f
        self.E_m = E_m
        self.G_f = G_f
        self.G_m = G_m
        self.nu_f = nu_f
        self.nu_m = nu_m
        self.l_f = l_f
        self.r_f = r_f
        self.vol_f = vol_f
        self.package = package
        self.get_effective_parameters()
        
    def get_effective_parameters(self):
        '''
        Calculates the effective parameters for given constituent parameters.

        Raises
        ------
        ValueError
            Package can only be "hex" or "square".

        Returns
        -------
        None.
        '''
        if self.package != 'hex' and self.package != 'square':
            raise ValueError('Package must be either "hex" or "square"!')
            
        if self.package == 'hex':
            p = 1/2*np.log(2*np.pi/(np.sqrt(3)*self.vol_f))
        else:
            p = 1/2*np.log(np.pi/self.vol_f)
            
        beta = np.sqrt(2*np.pi*self.G_m/(self.E_f*(np.pi*self.r_f**2)*p))
        nu1 = (self.E_f/self.E_m-1)/(self.E_f/self.E_m+2)
        nu2 = (self.G_f/self.G_m-1)/(self.G_f/self.G_m+1)
        
        self.E11 = self.E_f*(1-tanh(beta*self.l_f/2)/(beta*self.l_f/2))*self.vol_f\
            + self.E_m*(1-self.vol_f)
        self.E22 = self.E_m*(1+2*nu1*self.vol_f)/(1-nu1*self.vol_f)
        self.G12 = self.G_m*(1+2*nu2*self.vol_f)/(1-nu2*self.vol_f)
        self.nu12 = self.nu_f*self.vol_f + self.nu_m*(1-self.vol_f)
        self.nu21 = self.nu12*self.E22/self.E11

    def get_E(self, omega):
        '''
        Return Young's modulus as a function of angle omega
    
        Parameters
        ----------
        omega : float
            Angle of orientation in radians.
        
        Returns
        -------
        E : float
            Young's modulus in angle direction
        '''
        E = 1/(cos(omega)**4/self.E11+sin(omega)**4/self.E22\
               +1/4*(1/self.G12-2*self.nu12/self.E11)*sin(2*omega)**2)
        return E
    
    def get_accumulated_E(self, orientations):
        '''
        Return accumulated Young's modulus

        Parameters
        ----------
        orientations : dict of type float as in {angle, vol_frac}
            Volume fractions of discrete angles.

        Returns
        -------
        E_acc : float
            Effective Young's modulus.
        '''
        frac_acc = 0
        E_acc = 0
        for angle, frac in orientations:
            frac_acc = frac
            E = self.get_E(angle)
            E_acc += E*frac
            
        if frac_acc != 1:
            warnings.warn("The accumulated volume fraction is not equal to 1")
        return E_acc
            
    @staticmethod
    def turn_by_angle(Es, angle):
        l = len(Es)
        angle_frac = int(angle/360*l)
        Es_copy = Es.copy()
        Es_copy[0:angle_frac] = Es[-angle_frac:].copy()
        Es_copy[angle_frac:] = Es[:-angle_frac].copy()
        return Es_copy
    
    
    
if __name__ == "__main__":
    #%% Testing
    TH_Carb = Tsai_Hill(242*1e9, 1.18*1e9, 105*1e9, 0.4*1e9, 0.1, 0.35, 2.5*1e-3,
                    7.2/2*1e-6, 0.25)
    TH_Glass = Tsai_Hill(80*1e9, 1.18*1e9, 33*1e9, 0.4*1e9, 0.22, 0.35, 3.6*1e-3,
                    16/2*1e-6, 0.25)
    E_Glass = TH_Glass.get_E(0)*1e-9
    E_Carb = TH_Carb.get_E(0)*1e-9
    
    
    #%% Polar plot
    angles = np.arange(0,2*np.pi,0.001)
    Es_Glass = TH_Glass.get_E(angles)
    Es_Carb = TH_Carb.get_E(angles)
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angles, Es_Glass, label='HT Glass')
    ax.plot(angles, Es_Carb, label='HT Carbon')
    ax.plot(angles, 0.5*(Es_Carb+Es_Glass), label='HT Hybrid')
    ax.legend()
    #ax.set_rmax(2)
    #ax.set_rticks([0.5*1e10, 1*1e10, 1.5*1e10, 2*1e10])  # Less radial ticks
    #ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    
    ax.set_title("Young's modulus of fiber reinforced PA6", va='bottom')
    plt.show()
    
    #%% With orientations
    angle = 45
    angle_frac = int(angle/360*len(Es_Carb))
    
    # orientations = {0:0.7,20:0.2,45:0.1}    
    Es_Carb2 = Tsai_Hill.turn_by_angle(Es_Carb, angle)
    Es_Carb3 = Tsai_Hill.turn_by_angle(Es_Carb, 2*angle)
    Es_Carb4 = Tsai_Hill.turn_by_angle(Es_Carb, 3*angle)
    
    angles = np.arange(0,2*np.pi,0.001)
    Es_Glass = TH_Glass.get_E(angles)
    Es_Carb = TH_Carb.get_E(angles)
    
    #%%
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angles, Es_Carb, label='Carbon {}°'.format(angle*1))
    ax.plot(angles, Es_Carb2, label='Carbon {}°'.format(angle*2))
    ax.plot(angles, Es_Carb3, label='Carbon {}°'.format(angle*3))
    ax.plot(angles, Es_Carb4, label='Carbon {}°'.format(angle*4))
    ax.plot(angles, (0.25*Es_Carb+0.25*Es_Carb2+0.25*Es_Carb3+0.25*Es_Carb4), 
            linewidth = 3, label='Carbon Homogenized')
    ax.legend()
    #ax.set_rmax(2)
    #ax.set_rticks([0.5*1e10, 1*1e10, 1.5*1e10, 2*1e10])  # Less radial ticks
    #ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    
    ax.set_title("Young's modulus of fiber reinforced PA6", va='bottom')
    plt.show()


    #%% Isotropic plot
    n = 31
    angles = np.arange(0,2*np.pi,0.001)
    degrees = np.linspace(180./(n+1),180-180./(n+1),n)
    Es = np.zeros(len(Es_Carb))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    frac = 1./(len(degrees)+1)
    Es = frac*Es_Carb.copy()
    ax.plot(angles, Es_Carb, label='Carbon {}°'.format(0), 
                linewidth= 0.5, alpha=0.5)
    for degree in degrees:
        E_new = Tsai_Hill.turn_by_angle(Es_Carb, degree)
        Es += frac*E_new
        ax.plot(angles, E_new, label='Carbon {}°'.format(angle*1), 
                linewidth= 0.5, alpha=0.5)
    ax.plot(angles, Es, label='Carbon Homogenized',
            linewidth=3, color = '#20b2aa')
    ax.grid(True)
    ax.set_title("Young's modulus of fiber reinforced PA6", va='bottom')
    plt.show()

