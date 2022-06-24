# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:24:54 2022

@author: chri

3D and Polar plot of Young's modulus body based on Böhlke, Brüggemann (2001)
"""

import numpy as np

USEVOIGT = False
sin = np.sin
cos = np.cos

def diade(di, dj):
    '''
    Return diadic product of two direction vectors
    '''
    return np.einsum("i,j->ij", di, dj)

def matrix2voigt(matrix):
    '''
    Return the voigt notation of a symmetric 3x3 matrix
    '''
    return np.array([matrix[0,0], matrix[1,1], matrix[2,2],
                     matrix[1,2], matrix[0,2], matrix[0,1]])

def matrix2mandel(matrix):
    '''
    Return the mandel notation of a symmetric 3x3 matrix
    '''
    b = np.sqrt(2)
    return np.array([matrix[0,0], matrix[1,1], matrix[2,2],
                     b*matrix[1,2], b*matrix[0,2], b*matrix[0,1]])

def matrixreduction(matrix):
    if USEVOIGT:
        return matrix2voigt(matrix)
    else:
        return matrix2mandel(matrix)

def get_reciprocal_E(didi, S):
    '''
    Return the reciprocal of Young's modulus
    '''
    return np.einsum('i,ij,j->', didi, S, didi)

def get_E(di, S):
    '''
    Return Young's modulus
    '''
    didi = matrixreduction(diade(di, di))
    return get_reciprocal_E(didi, S)**(-1)

def dir_vec(theta, phi):
    '''
    Return direction vector (parameterized with angles theta and phi)
    '''
    return np.array([cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi)])


def plot_E_body(S, o, p, bound=[0,0,0]):
    n = int(o); m = int(p)
    E_x = np.zeros((n+1, m+1))
    E_y = np.zeros((n+1, m+1))
    E_z = np.zeros((n+1, m+1))
    for i in range(0, n+1, 1):
        for j in range(0, m+1, 1):
            theta = i/n*2*np.pi
            phi = j/m*np.pi
            vec = dir_vec(theta, phi)
            E = get_E(vec, S)
            E_x[i, j] = E*vec[0]
            E_y[i, j] = E*vec[1]
            E_z[i, j] = E*vec[2]
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    x = E_x; y = E_y; z = E_z
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.set_xlabel('E11')
    ax.set_ylabel('E22')
    ax.set_zlabel('E33')
    

    ax.plot_surface(x, y, z, 
                    cmap=cm.viridis,)
    
    if not bound[0]==0:
        ax.set_xlim(-bound[0],bound[0])
    if not bound[1]==0:
        ax.set_ylim(-bound[1],bound[1])
    if not bound[2]==0:
        ax.set_zlim(-bound[2],bound[2])

    plt.show()
    
def polar_plot_E_body(S, o, angle, bound=[0,0,0], plot=True):
    n = int(o)
    E = np.zeros(n+1)
    rad = np.zeros(n+1)
    
    phi = np.pi/2 + angle #changing angle does not work yet
    for i in range(0, n+1, 1):
        theta = i/n*2*np.pi
        vec = dir_vec(theta, phi)
        E_temp = get_E(vec, S)
        E[i] = E_temp
        rad[i] = i/n*2*np.pi
        
    if plot == True:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(rad, E)
        #ax.set_rmax(2)
        #ax.set_rticks([0.5*1e10, 1*1e10, 1.5*1e10, 2*1e10])  # Less radial ticks
        #ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        
        ax.set_title("Young's modulus over angle", va='bottom')
        plt.show()
    else:
        return rad, E
    
def polar_plot(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for datum in data:
        try:
            ax.plot(datum[0], datum[1], label=datum[2])
        except:
            ax.plot(datum[0], datum[1])
    ax.grid(True)
    ax.set_title("Young's modulus over angle", va='bottom')
    ax.legend()
    plt.show()
    
    

    