# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:24:53 2023

@author: User
"""

import numpy as np
from scipy.optimize import nnls
from scipy.signal import sosfiltfilt, butter
from numpy import tan as tan

## matrix A. it only depends on the grid and particle structure, independent on the cld.
def get_A(grid, cir:"aspect ratio")->"array of shape (M x N)":
    ## probability funtion for a sphere
    def pdf(L_l, L_u, a_i):
        if (L_u <= L_l):
            raise ValueError ("the uuper bound must be larger than the lower bound")

        if (L_u <= 2.0*a_i):
            return np.sqrt(1.0-(L_l/2.0/a_i)**2) - np.sqrt(1.0-(L_u/2.0/a_i)**2)
        elif (L_l> 2.0*a_i):
            return 0
        else:
            return np.sqrt(1.0-(L_l/2.0/a_i)**2)
        
    ## probability funtion for an ellipse
    def pdf_ellip(L_l, L_u, a, a_i, cir):
        if (L_u <= L_l):
            raise ValueError ("the upper bound must be larger than the lower bound")

        if (a == np.pi) or (a == 0.0):
            if (L_u <= 2.0*a_i):
                return np.sqrt(1.0-(L_l/2.0/a_i)**2) - np.sqrt(1.0-(L_u/2.0/a_i)**2)
            elif (L_l> 2.0*a_i):
                return 0
            else:
                return np.sqrt(1.0-(L_l/2.0/a_i)**2)

        elif (a == np.pi*0.5) or (a == np.pi*1.5):
            if (L_u <= 2.0*cir*a_i):
                return np.sqrt(1.0-(L_l/2.0/a_i/cir)**2) - np.sqrt(1.0-(L_u/2.0/a_i/cir)**2)
            elif (L_l> 2.0*cir*a_i):
                return 0
            else:
                return np.sqrt(1.0-(L_l/2.0/a_i/cir)**2)

        else:
            if 2.0*cir*a_i*np.sqrt((1+tan(a)**2) / (cir**2 + tan(a)**2)) >= L_u:
                return np.sqrt(1.0-(cir**2 + tan(a)**2)/(1.0+tan(a)**2) * (L_l/2.0/cir/a_i)**2) - np.sqrt(1.0-(cir**2 + tan(a)**2)/(1.0+tan(a)**2) * (L_u/2.0/cir/a_i)**2)
            elif 2.0*cir*a_i*np.sqrt((1+tan(a)**2) / (cir**2 + tan(a)**2)) < L_l:
                return 0
            else:
                return np.sqrt(1.0-(cir**2 + tan(a)**2)/(1.0+tan(a)**2) * (L_l/2.0/cir/a_i)**2) 
            
    # grid 
    def get_centers(grid):
        ct = []
        for p in range(1, len(grid)):
            ct.append((grid[p] + grid[p-1]) / 2)
        ct = np.asarray(ct)
        return ct
        
    ## bin number of chords M and particle count N, assume they use the same grid, but can be changed later
    M = len(grid) - 1
    N = len(grid) - 1

    ## chrod length grid
    l_ct = get_centers(grid)
    l_grid = grid

    ## particle size grid
    x_ct = get_centers(grid)
    x_grid = grid
    
    ## format A
    if cir == 1.0:
        A = []
        for j in range (0, M):
            for i in range (0, N):
                A.append(pdf(l_grid[j], l_grid[j+1], x_ct[i]/2.0))
        A = np.asarray(A)
        A = np.reshape(A, (M,N))
        
    elif cir < 1.0 and cir > 0.0:    
        A = []
        itg_res = 100  ## this can be changed, usually 100 is enough
        pi_space = np.linspace(0, 2.0*np.pi, itg_res)
        for j in range (0, M):
            for i in range (0, N):
                area_wrapper = []
                for k in range (0, itg_res):
                    a = pi_space[k]
                    area_wrapper.append(pdf_ellip(x_grid[j], x_grid[j+1], a, x_ct[i]/2.0/np.sqrt(cir), cir))  # numerical integration
                A.append(np.trapz(area_wrapper, pi_space) / 2.0/np.pi)
        A = np.asarray(A)
        A = np.reshape(A, (M,N))
        
    else:
        raise ValueError ("the cir must be between 0 and 1.")
    
    return A


## cld to psd
def cld_to_psd(A, cld:"normlaized cld", fre:"frequency of Butterworth"=0.35, order:"order of Butterworth"=1)->"normalized filtered psd, res" :       
    ## inverse cld to get psd
    X_inverse, residuals = nnls(A, cld)

    ## the order and frequency can be adjusted
    sos = butter(order, fre, output='sos')  # default is lowpass
    filtd_X = sosfiltfilt(sos, X_inverse)
    
    return filtd_X, residuals