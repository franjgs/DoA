#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:52:17 2019

@author: fran
"""
import numpy as np
from math import cos, sin, sqrt
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.optimize import linear_sum_assignment

import pickle

from scipy.signal import find_peaks

# Functions

def sph2V(r, theta, phi):
    """ Theta-phi to U-V direction cosines
  
    Args:
        theta (float or np.array): Theta angle, in radians
        phi (float or np.array): Phi angle, in radians
  
    Returns:
      (u, v): Tuple of corresponding (u, v) direction cosines
    """
    x = r*np.sin(theta) * np.cos(phi)
    y = r*np.sin(theta) * np.sin(phi)
    z = r*np.cos(theta)
    
    U=Vector(x,y,z)
    return U

def uv_to_thetaphi(u, v):
    """ U-V direction cosines to theta-phi coordinates
  
    Args:
        u (float or np.array): U direction cosine
        v (float or np.array): V direction cosine
  
    Returns:
      (theta, phi): Tuple of corresponding (theta, phi) angles, in radians
    """ 
    theta = np.arcsin(np.sqrt(u**2 + v**2))
    phi = np.arctan2(u, v)
    
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    
    return theta, phi
    
def uv_to_azel(u, v):
    """ U-V direction cosines to azimuth-elevation coordinates
  
    Args:
        u (float or np.array): U direction cosine
        v (float or np.array): V direction cosine
  
    Returns:
      (az, el): Tuple of corresponding (azimuth, elevation) angles, in radians
    """ 
    az = np.arctan2(u, np.sqrt(1 - u**2 - v**2))
    el = np.arcsin(v)
    
    return az, el

def sph2cart1(r, th, phi):
    x = r * cos(phi) * sin(th)
    y = r * sin(phi) * sin(th)
    z = r * cos(th)

    return x, y, z
  
def cart2sph1(x, y, z):
    r = sqrt(x**2 + y**2 + z**2) + 1e-15
    th = np.acos(z / r)
    phi = np.atan2(y, x)

    return r, th, phi

def V2sph1(vector):
    x=vector.coord[0]
    y=vector.coord[1]
    z=vector.coord[2]
    
    r = sqrt(x**2 + y**2 + z**2) + 1e-15
    th = np.acos(z / r)
    phi = np.atan2(y, x)
    
    return r, th, phi

def V2A(vector):
    x=vector.coord[0]
    y=vector.coord[1]
    z=vector.coord[2]
    
    r = sqrt(x**2 + y**2 + z**2) + 1e-15
    th = np.acos(z / r)
    phi = np.atan2(y, x)
    
    A=Angle(th,phi)
    return A

def A2V(angle):
    th = angle.angle[0]
    phi = angle.angle[1]
    x = cos(phi) * sin(th)
    y = sin(phi) * sin(th)
    z = cos(th)

    V=Vector(x,y,z)
    return V


class Vector:
  def __init__(self, x, y, z):
    self.coord = np.array([x,y,z])

class Angle:
  def __init__(self, theta, phi):
    self.angle = np.array([theta,phi])

#class Matrix(object):
#    def __init__(self, signal, array):
#        self.data=np.zeros([Array.M,signal.D])
#        kappa=array.k
#        for d in range(signal.D):
#            k=-A2V(signal.angles[d]).coord
#            for m in range(Array.M):
#                rm=Array.rm[m]
#                rm_dot_k=np.dot(rm,k)
#                self.data[m,d]=signal.amp[d]*np.exp(1j*kappa*rm_dot_k)*array.g_rad[m](signal.angles[d].angle[0],signal.angles[d].angle[1])

def gISO(theta,phi):
    return 1

def gHWDip(theta,phi):
    """
    Fichero: Arrays.pdf, 22.3. Array Pattern Multiplication 1095
    """
    num=np.cos(0.5*np.pi*np.cos(theta))
    den=np.sin(theta)
    g=(num/den)**2
    if num.size>1:
        g[np.where(abs(theta)<np.finfo(float).eps)[0]]=0
    else:
        if abs(theta)<np.finfo(float).eps:
            g=0
    return g

g_rad={'ISO': gISO, 'HWDip': gHWDip}

class Array(object):
    def __init__(self, M, rm, am, kappa, ant_type):
        self.M=M
        self.rm=rm
        # self.Delta = rm[1]-rm[0]
        self.am = am
        self.k = kappa
        # self.kappaDelta = self.k * self.Delta
        self.g_rad=[g_rad[ant_type] for kk in range(M)]

class Signal(object):
    def __init__(self, D, Thetas, Phis, Amps):
        self.D=D
        self.angles=[Angle(Thetas[xm],Phis[xm]) for xm in range(D)]
        self.amps=Amps

class Matrix(object):
    def __init__(self, signal, array):
        self.data=np.zeros([array.M,signal.D]) + 1j*np.zeros([array.M,signal.D])
        kappa=array.k
        for d in range(signal.D):
            k=A2V(signal.angles[d]).coord
            for m in range(array.M):
                rm=array.rm[m]
                rm_dot_k=np.dot(rm,k)
                self.data[m,d]=signal.amps[d]*np.exp(1j*kappa*rm_dot_k)*array.g_rad[m](signal.angles[d].angle[0],signal.angles[d].angle[1])

class Matrix2:
    def __init__(self, *args, **kwargs):
        self.args=dict(**kwargs)
        for attr in kwargs.keys():
            self.__dict__[attr] = kwargs[attr]
        
        self.angles=args[0]
        self.amps=args[1]
        self.D=len(self.angles)
        self.M=len(self.rm)
        # self.Delta = self.rm[1]-self.rm[0]
        # kwargs ={'k': 2*np.pi, 'g_rad': [g_rad['ISO'] for kk in range(self.M)], **kwargs}
        kwargs ={'k': 2*np.pi, 'g_rad': 'ISO', **kwargs}
        self.k=kwargs['k']
        # self.kappaDelta = self.k * self.Delta
        self.g_rad=[g_rad[kwargs['g_rad']] for kk in range(self.M)] 
        self.data=np.zeros([self.M,self.D])+ 1j*np.zeros([self.M,self.D])

        for d in range(self.D):
            k=A2V(self.angles[d]).coord
            for m in range(self.M):
                rm=self.rm[m]
                rm_dot_k=np.dot(rm,k)
                self.data[m,d]=self.amps[d]*np.exp(1j*self.k*rm_dot_k)*self.g_rad[m](self.angles[d].angle[0],self.angles[d].angle[1])
                 
# rd = np.matrix([np.linspace(0,(M-1)/2,M),np.zeros(M),np.zeros(M)])


def ArrayFactor(array,theta,phi):
    """ Generalized Array Factor
    """
    kappa=array.k
    k=sph2V(1, theta, phi).coord
    v=0
    for m in range(array.M):
        rm=Array.rm[m]
        rm_dot_k=np.dot(rm,k)
        v += array.am[m]*np.exp(1j*kappa*rm_dot_k)
    return v

def ArrayGain(array,theta,phi):
    """ Generalized Array Gain
    Includes the Radiation Pattern of each element
    """
    kappa=array.k
    k=sph2V(1, theta, phi).coord
    v=0
    for m in range(array.M):
        rm=array.rm[m]
        rm_dot_k=np.dot(rm,k)
        v += array.am[m]*np.exp(1j*kappa*rm_dot_k)*array.g_rad[m](theta,phi)
    return v

def ArrayResponse(array,theta,phi):
    """ Generalized Array Gain
    Includes the Radiation Pattern of each element
    """
    kappa=array.k
    k=sph2V(1, theta, phi).coord
    v=np.zeros([array.M,1])+1j*np.zeros([array.M,1])
    for m in range(array.M):
        rm=array.rm[m]
        rm_dot_k=np.dot(rm,k)
        v[m]= array.am[m]*np.exp(1j*kappa*rm_dot_k)*array.g_rad[m](theta,phi)
    return v

# af00=ArrayFactor(Array,0,0)
# ag00=ArrayGain(Array,0,0)


def ArrayEHPlanePlot(Array, isLog=True):
    """
    Plot 2D plots showing E-field for E-plane (phi = 0°) and the H-plane (phi = 90°).
    """

    Xtheta = np.linspace(0, 180, 180)                    # Theta range array used for plotting

    if isLog:                                          # Can plot the log scale or normal
        plt.plot(Xtheta, 10 * np.log10(abs(ArrayGain(Array,np.pi*Xtheta/180,np.radians(90)))), label="H-plane (Phi=90°)")  # Log = 20 * log10(E-field)
        plt.plot(Xtheta, 10 * np.log10(abs(ArrayGain(Array,np.pi*Xtheta/180,0))), label="E-plane (Phi=0°)")
        plt.ylabel('E-Field (dB)')
    else:
        plt.plot(Xtheta, (abs(ArrayGain(Array,np.pi*Xtheta/180,np.radians(90)))), label="H-plane (Phi=90°)")    # Log = 20 * log10(E-field)
        plt.plot(Xtheta, (abs(ArrayGain(Array,np.pi*Xtheta/180,0))), label="E-plane (Phi=0°)")
        plt.ylabel('E-Field')

    plt.xlabel('Theta (degs)')                 # Plot formatting
    plt.title("ArrayGain: M=" + str(Array.M) + " g_rad=" + Array.g_rad[0].__name__)
    plt.ylim(-40)
    plt.xlim((0, 180))

    start, end = plt.xlim()
    plt.xticks(np.arange(start, end, 10))
    plt.grid(b=True, which='major')
    plt.legend()
    plt.show()                                 # Show plot
                                               # Return the calculated fields

def ArrayGainHPlot(Array):
    # normalize and convert to dB
    dbnorm = lambda x: 10*np.log10(np.abs(x)/np.max(np.abs(x)));

    # generate example data
    # some angles
    alpha = np.arange(0, 180, 0.01);
    x = np.deg2rad(alpha)
    
    Array_H=ArrayGain(Array,np.radians(90),x)
    dir_function = dbnorm(Array_H)
    
    # plot
    ax = plt.subplot(111, polar=True)
    # set zero north
    ax.set_theta_zero_location('E')
    ax.set_theta_direction('counterclockwise')
    plt.plot(np.deg2rad(alpha), dir_function)
    ax.set_ylim(-20,0)
    ax.set_yticks(np.array([-20, -12, -6, 0]))
    # ax.set_xticks(np.pi*np.array([0, -45, -90, np.nan, np.nan, np.nan, 90, 45])/180)
    
    # The new way per https://github.com/matplotlib/matplotlib/pull/4699
    ax.set_xticks(np.pi*np.array([0, 45, 90, 135, 180])/180)
    ax.set_thetalim(0, np.pi)

    plt.title("ArrayGain (H-Plane): M=" + str(Array.M) + " g_rad=" + Array.g_rad[0].__name__)
    plt.xlabel(r'$\phi$ (degs)')                 # Plot formatting


    # Or if you still want a full 360 radiation pattern uncomment these
    # ax.set_xticks(np.pi*np.array([-90, -45, 0, 45, 90])/180)
    # ax.set_thetalim(-np.pi, np.pi)


    plt.show()

def ArrayGainEPlot(Array):
    # E-plane or elevation angle
    # normalize and convert to dB
    dbnorm = lambda x: 10*np.log10(np.abs(x)/np.max(np.abs(x)));

    # generate example data
    # some angles
    alpha = np.arange(0, 180, 0.01);
    x = np.deg2rad(alpha)
    
    Array_E=ArrayGain(Array,x,np.radians(0))
    dir_function = dbnorm(Array_E)
    
    # plot
    ax = plt.subplot(111, polar=True)
    # set zero north
    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    plt.plot(np.deg2rad(alpha), dir_function)
    ax.set_ylim(-20,0)
    ax.set_yticks(np.array([-20, -12, -6, 0]))
    # ax.set_xticks(np.pi*np.array([0, -45, -90, np.nan, np.nan, np.nan, 90, 45])/180)
    
    # The new way per https://github.com/matplotlib/matplotlib/pull/4699
    ax.set_xticks(np.pi*np.array([0, 45, 90, 135, 180])/180)
    ax.set_thetalim(0, np.pi)
    plt.title("ArrayGain (E-Plane): M=" + str(Array.M) + " g_rad=" + Array.g_rad[0].__name__)
    plt.xlabel(r'$\theta$ (degs)')                 # Plot formatting


    # Or if you still want a full 360 radiation pattern uncomment these
    # ax.set_xticks(np.pi*np.array([-90, -45, 0, 45, 90])/180)
    # ax.set_thetalim(-np.pi, np.pi)


    plt.show()
    
def SurfacePlot(Array):
    """Plots 3D surface plot over given theta/phi range in Fields by calculating cartesian coordinate equivalent of spherical form."""

    print("Processing SurfacePlot...")

    # dbnorm = lambda x: 10*np.log10(np.abs(x)/np.max(np.abs(x)));

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)
    ax.axis('on')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticklabels([]) 
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    phi, theta = np.linspace(0, 2*np.pi, 80), np.linspace(0,np.pi, 80)
    PHI, THETA = np.meshgrid(phi,theta)

    R=np.empty(PHI.shape)
    #colors =plt.cm.jet( (X.max()-X)/float((X-X.min()).max()))
    
    th_c=0
    for thk in theta:
        ph_c=0
        for phk in phi:
            R[th_c,ph_c]=np.abs(ArrayGain(Array,thk,phk))
            ph_c+=1
        th_c+=1    
    
 
    # R=dbnorm(R)
    # Rmax=np.max(R)
    # R=R-np.min(R)
    
    X = R*np.sin(THETA) * np.cos(PHI)
    Y = R*np.sin(THETA) * np.sin(PHI)
    Z = R*np.cos(THETA)
    
    # colors =plt.cm.jet( (R)/(Rmax) )
    colors =plt.cm.jet(R/np.max(R))
    ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, facecolors=colors,
            linewidth=0, antialiased=True, alpha=0.5, zorder = 0.5)

    ax.view_init(azim=45, elev = 10)
    # ax.view_init(azim=0, elev = 0)

    # Add Spherical Grid
    R = np.max(R)
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    ax.plot_wireframe(X, Y, Z, linewidth=0.5, rstride=20, cstride=20)


    x_sens=[Array.rm[mm][0] for mm in range(Array.M)]
    y_sens=[Array.rm[mm][1] for mm in range(Array.M)]
    z_sens=[Array.rm[mm][2] for mm in range(Array.M)]
    
    ax.plot(x_sens,y_sens,z_sens,'^')

    #  plt.show()

    ################################
    # phi,theta=np.mgrid[0:2*np.pi:201j, 0:np.pi:101j]
    # ze = np.abs(ArrayGain(Array,theta,phi))

    # ax.plot_surface(X, Y, Z, color='b')                                          # Plot surface
    # plt.ylabel('Y')
    # plt.xlabel('X')                                                              # Plot formatting
    ################################
    plt.title("ArrayGain: M=" + str(Array.M) + " g_rad=" + Array.g_rad[0].__name__)
    # plt.legend()

    plt.show()

# Subspace Projection
def SubProj(CovMat,**kwargs):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    Lamb,V = LA.eig(CovMat)

    idx=np.abs(Lamb).argsort()[::-1]
    Lamb=Lamb[idx]
    V=V[:,idx]

    if 'L' in kwargs:
        L=kwargs['L']
    else:
        PoVk=np.cumsum(np.abs(Lamb))/np.sum(np.abs(Lamb))
        if 'PoV' in kwargs:
            PoV=kwargs['PoV']
        else:    
            PoV=0.995
        L=np.where(PoVk>PoV)[0][0]
    ES = V[:,0:L]
    
#    S = U[:,0:L]
#    Phi = LA.pinv(S[0:N-1]) @ S[1:N] # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
#    eigs,_ = LA.eig(Phi)
#    DoAsESPRIT = np.arcsin(np.angle(eigs)/np.pi)
#    return DoAsESPRIT

    return ES
    
# def fmin(x, T, GES, array, G,Gt):
def fmin1D(x, *args1):

    D=args1[0]

    nc=0
    thetas=x[nc*D:(nc+1)*D]
    nc+=1
    
    phis=0.0*np.pi*np.ones(D)
    # phis=x[nc*D:(nc+1)*D]
    # nc+=1
    
    amps=np.ones(D)
    # amps=x[nc*D:(nc+1)*D]
    # nc+=1
    
    if len(args1)==6:
        T=args1[1]
        GES=args1[2]
        array=args1[3]
        G=args1[4]
        Gt=args1[5]

    elif len(args1)==5:
        T=np.zeros((D,D))+1j*np.zeros((D,D))
        
        for d in range(D):
            T[:][d] = x[nc*D:(nc+1)*D] + 1j * x[(nc+D)*D:(nc+D+1)*D]
            nc+=1
        GES=args1[1]
        array=args1[2]
        G=args1[3]
        Gt=args1[4]
                
    angles=[Angle(thetas[xm],phis[xm]) for xm in range(D)]
    A=Matrix2(angles,amps,rm=array.rm).data
    kappa=array.k
    Gtrm=np.matmul(Gt,np.array(array.rm))
    Gtrm_k=np.empty((Gt.shape[0],D))
    
    for d in range(D):
        k=sph2V(1, thetas[d], phis[d]).coord
        Gtrm_k[:,d]=np.matmul(Gtrm,k)
    B=np.exp(1j*kappa*Gtrm_k)
    GA=G@A
    if B.shape[0] != GA.shape[0]:
        GA_tile=np.tile(GA,(Gt.shape[0]//array.M,1))
        GAB=GA_tile*B
    else:
        GAB=GA*B
    # T=np.matmul(LA.pinv(GAB),GES) #  np.eye(Signal.D)    

    fx=LA.norm(GES-GAB@T,'fro')    
    # =LA.norm(GES-((G@A)*B)@T,'fro')   
    # It seems to be equivalent to 
    # err=GES-GAB@T
    # fx=LA.norm(np.concatenate((err.real.reshape(-1), err.imag.reshape(-1))),2)
    
    return fx

def fmin1D1(x, *args1):

    D=args1[0]

    nc=0
    thetas=x[nc*D:(nc+1)*D]
    nc+=1
    
    phis=0.0*np.pi*np.ones(D)
    # phis=x[nc*D:(nc+1)*D]
    # nc+=1
    
    amps=np.ones(D)
    # amps=x[nc*D:(nc+1)*D]
    # nc+=1
    
    if len(args1)==6:
        T=args1[1]
        GES=args1[2]
        array=args1[3]
        G=args1[4]
        Gt=args1[5]

    elif len(args1)==5:
        T=np.zeros((D,D))+1j*np.zeros((D,D))
        
        for d in range(D):
            T[:][d] = x[nc*D:(nc+1)*D] + 1j * x[(nc+D)*D:(nc+D+1)*D]
            nc+=1
        GES=args1[1]
        array=args1[2]
        G=args1[3]
        Gt=args1[4]
                
    angles=[Angle(thetas[xm],phis[xm]) for xm in range(D)]
    A=Matrix2(angles,amps,rm=array.rm).data
    kappa=array.k
    Gtrm=np.matmul(Gt,np.array(array.rm))
    Gtrm_k=np.empty((Gt.shape[0],D))
    
    for d in range(D):
        k=sph2V(1, thetas[d], phis[d]).coord
        Gtrm_k[:,d]=np.matmul(Gtrm,k)
    B=np.exp(1j*kappa*Gtrm_k)
    GA=G@A
    if B.shape[0] != GA.shape[0]:
        GA_tile=np.tile(GA,(Gt.shape[0]//array.M,1))
        GAB=GA_tile*B
    else:
        GAB=GA*B
    # T=np.matmul(LA.pinv(GAB),GES) #  np.eye(Signal.D)    

    fx=LA.norm(GES-GAB@T,ord=1)    
    # =LA.norm(GES-((G@A)*B)@T,'fro')    
    return fx

def fmin1D2(x, *args1):

    D=args1[0]

    nc=0
    thetas=x[nc*D:(nc+1)*D]
    nc+=1
    
    phis=0.0*np.pi*np.ones(D)
    # phis=x[nc*D:(nc+1)*D]
    # nc+=1
    
    amps=np.ones(D)
    # amps=x[nc*D:(nc+1)*D]
    # nc+=1
    
    if len(args1)==6:
        T=args1[1]
        GES=args1[2]
        array=args1[3]
        G=args1[4]
        Gt=args1[5]

    elif len(args1)==5:
        T=np.zeros((D,D))+1j*np.zeros((D,D))
        
        for d in range(D):
            T[:][d] = x[nc*D:(nc+1)*D] + 1j * x[(nc+D)*D:(nc+D+1)*D]
            nc+=1
        GES=args1[1]
        array=args1[2]
        G=args1[3]
        Gt=args1[4]
                
    angles=[Angle(thetas[xm],phis[xm]) for xm in range(D)]
    A=Matrix2(angles,amps,rm=array.rm).data
    kappa=array.k
    Gtrm=np.matmul(Gt,np.array(array.rm))
    Gtrm_k=np.empty((Gt.shape[0],D))
    
    for d in range(D):
        k=sph2V(1, thetas[d], phis[d]).coord
        Gtrm_k[:,d]=np.matmul(Gtrm,k)
    B=np.exp(1j*kappa*Gtrm_k)
    GA=G@A
    if B.shape[0] != GA.shape[0]:
        GA_tile=np.tile(GA,(Gt.shape[0]//array.M,1))
        GAB=GA_tile*B
    else:
        GAB=GA*B
    # T=np.matmul(LA.pinv(GAB),GES) #  np.eye(Signal.D)    

    fx=LA.norm(GES-GAB@T,ord=2)    
    # =LA.norm(GES-((G@A)*B)@T,'fro')    
    return fx


def fmin1D_lsq(x, *args1):

    D=args1[0]

    nc=0
    thetas=x[nc*D:(nc+1)*D]
    nc+=1
    
    phis=0.5*np.pi*np.ones(D)
    # phis=x[nc*D:(nc+1)*D]
    # nc+=1
    
    amps=np.ones(D)
    # amps=x[nc*D:(nc+1)*D]
    # nc+=1
    
    if len(args1)==6:
        T=args1[1]
        GES=args1[2]
        array=args1[3]
        G=args1[4]
        Gt=args1[5]

    elif len(args1)==5:
        T=np.zeros((D,D))+1j*np.zeros((D,D))
        
        for d in range(D):
            T[:][d] = x[nc*D:(nc+1)*D] + 1j * x[(nc+D)*D:(nc+D+1)*D]
            nc+=1
        GES=args1[1]
        array=args1[2]
        G=args1[3]
        Gt=args1[4]
                
    angles=[Angle(thetas[xm],phis[xm]) for xm in range(D)]
    A=Matrix2(angles,amps,rm=array.rm).data
    kappa=array.k
    Gtrm=np.matmul(Gt,np.array(array.rm))
    Gtrm_k=np.empty((Gt.shape[0],D))
    
    for d in range(D):
        k=sph2V(1, thetas[d], phis[d]).coord
        Gtrm_k[:,d]=np.matmul(Gtrm,k)
    B=np.exp(1j*kappa*Gtrm_k)
    GA=G@A
    if B.shape[0] != GA.shape[0]:
        GA_tile=np.tile(GA,(Gt.shape[0]//array.M,1))
        GAB=GA_tile*B
    else:
        GAB=GA*B
    # T=np.matmul(LA.pinv(GAB),GES) #  np.eye(Signal.D)    

    err=GES-GAB@T
    fx=np.concatenate((err.real.reshape(-1), err.imag.reshape(-1)))
    # =LA.norm(GES-((G@A)*B)@T,'fro')    
    return fx

def fmin1D_noargs(x):
    PATH = '/content/gdrive/'+'MyDrive/Colab Notebooks/DoA/'
    with open(PATH+'train.pickle', 'rb') as f:
        args_ga = pickle.load(f)
        
    D=args_ga[0]

    nc=0
    thetas=x[nc*D:(nc+1)*D]
    nc+=1
    
    phis=0.5*np.pi*np.ones(D)
    # phis=x[nc*D:(nc+1)*D]
    # nc+=1
    
    amps=np.ones(D)
    # amps=x[nc*D:(nc+1)*D]
    # nc+=1
    
    if len(args_ga)==6:
        T=args_ga[1]
        GES=args_ga[2]
        array=args_ga[3]
        G=args_ga[4]
        Gt=args_ga[5]

    elif len(args_ga)==5:
        T=np.zeros((D,D))+1j*np.zeros((D,D))
        
        for d in range(D):
            T[:][d] = x[nc*D:(nc+1)*D] + 1j * x[(nc+D)*D:(nc+D+1)*D]
            nc+=1
        GES=args_ga[1]
        array=args_ga[2]
        G=args_ga[3]
        Gt=args_ga[4]
                
    angles=[Angle(thetas[xm],phis[xm]) for xm in range(D)]
    A=Matrix2(angles,amps,rm=array.rm).data
    kappa=array.k
    Gtrm=np.matmul(Gt,np.array(array.rm))
    Gtrm_k=np.empty((Gt.shape[0],D))
    
    for d in range(D):
        k=sph2V(1, thetas[d], phis[d]).coord
        Gtrm_k[:,d]=np.matmul(Gtrm,k)
    B=np.exp(1j*kappa*Gtrm_k)

    GA=G@A

    if B.shape[0] != GA.shape[0]:
        GA_tile=np.tile(GA,(Gt.shape[0]//array.M,1))
        GAB=GA_tile*B
    else:
        GAB=GA*B
    # T=np.matmul(LA.pinv(GAB),GES) #  np.eye(Signal.D)    

    err=GES-GAB@T
    fx=LA.norm(err,'fro')
    
    return fx

def fmin1Dint_noargs(x):
    PATH = '/content/gdrive/'+'MyDrive/Colab Notebooks/DoA/'
    with open(PATH+'train.pickle', 'rb') as f:
        args_ga = pickle.load(f)
    D=args_ga[0]
    
    nc=0
    thetas=np.radians(x[nc*D:(nc+1)*D])
    nc+=1
    phis=0.5*np.pi*np.ones(D)
    # phis=x[nc*D:(nc+1)*D]
    # nc+=1
    amps=np.ones(D)
    # amps=x[nc*D:(nc+1)*D]
    # nc+=1
    if len(args_ga)==6:
        T=args_ga[1]
        GES=args_ga[2]
        array=args_ga[3]
        G=args_ga[4]
        Gt=args_ga[5]
    elif len(args_ga)==5:
        T=np.zeros((D,D))+1j*np.zeros((D,D))
        for d in range(D):
            T[:][d] = x[nc*D:(nc+1)*D] + 1j * x[(nc+D)*D:(nc+D+1)*D]
            nc+=1
        GES=args_ga[1]
        array=args_ga[2]
        G=args_ga[3]
        Gt=args_ga[4]
    angles=[Angle(thetas[xm],phis[xm]) for xm in range(D)]
    A=Matrix2(angles,amps,rm=array.rm).data
    kappa=array.k
    Gtrm=np.matmul(Gt,np.array(array.rm))
    Gtrm_k=np.empty((Gt.shape[0],D))
    for d in range(D):
        k=sph2V(1, thetas[d], phis[d]).coord
        Gtrm_k[:,d]=np.matmul(Gtrm,k)
    B=np.exp(1j*kappa*Gtrm_k)
    GA=G@A
    if B.shape[0] != GA.shape[0]:
        GA_tile=np.tile(GA,(Gt.shape[0]//array.M,1))
        GAB=GA_tile*B
    else:
        GAB=GA*B
    err=GES-GAB@T
    fx=LA.norm(err,'fro')
    
    return fx
    
def match_vec(v1, v2, dist):
    assert v1.ndim == v2.ndim == 1
    assert v1.shape[0] == v2.shape[0]
    n = v1.shape[0]
    t = np.dtype(dist(v1[0], v2[0]))
    dist_matrix = np.fromiter((dist(x1, x2) for x1 in v1 for x2 in v2),
                              dtype=t, count=n*n).reshape(n, n)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return v2[col_ind]

def DoA_MVDR(CovMat,L,Array,Angles):
    # CovMat is the signal covariance matrix, 
    # L is the number of sources, 
    # Array is the structure than describes the array
    #   Array.M is the number of antennas
    #   Array.rm holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    
    numAngles = len(Angles)
    # pspectrum = np.zeros(numAngles,dtype=np.complex_)
    pspectrum = np.zeros(numAngles) 
    
    # signal_amp = 1 + 1j*0   # Signal amplitude(s) is (are) unknown
    CovMatInv= np.linalg.inv(CovMat)
    for i in range(numAngles):
        av = ArrayResponse(Array,Angles[i],0)
        con_av = np.conjugate(av.data)
        con_av_tp = con_av.transpose()
        
        pspectrum[i] = 1 / np.abs(con_av_tp @ CovMatInv @ av.data)

    DoAsMVDR,_= find_peaks(pspectrum,height=1.35, distance=1.5)
    return Angles[DoAsMVDR], pspectrum

def MUSIC(CovMat,L,Array,Angles):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    Lamb,V = LA.eig(CovMat)
        
    idx=np.abs(Lamb).argsort()[::-1]
    Lamb=Lamb[idx]
    V=V[:,idx]
    
    L = min(L,Array.M-1)
    
    Qn  = V[:,L:Array.M]
    numAngles = Angles.size
    pspectrum = np.zeros(numAngles)
    for i in range(numAngles):
        av = ArrayResponse(Array,Angles[i],0)
        pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@av))
    # psindB = np.log10(10*pspectrum/pspectrum.min())
    psindB = pspectrum
    DoAsMUSIC,_= find_peaks(psindB,height=1.35, distance=1.5)
    return Angles[DoAsMUSIC],pspectrum

def ESPRIT_1D(CovMat,L,N, kappaDelta):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # _,U = LA.eig(CovMat)
    # S = U[:,0:L]
    S = SubProj(CovMat,L=L)
    
    Phi = LA.pinv(S[0:N-1]) @ S[1:N]   # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs,_ = LA.eig(Phi)
    
    # Phi2 = LA.pinv(S[1:N]) @ S[0:N-1]   # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    # eigs2,_ = LA.eig(Phi2)
    """
    # DoAsESPRIT = np.pi-np.arccos(np.angle(eigs)/(2*np.pi*Delta))    
    DoAsESPRIT = np.pi-np.arccos(np.angle(eigs)/Delta) # (2*np.pi*Delta))
    print('\nDoAs (\phi) cos = ', np.degrees(DoAsESPRIT))
    DoAsESPRIT2 = np.pi-np.arccos(np.angle(eigs2)/Delta) # (2*np.pi*Delta))
    print('DoAs (\phi2) cos = ', np.degrees(DoAsESPRIT2))
    """
    # # DoAsESPRIT = 0.5*np.pi+np.arcsin(np.angle(eigs)/Delta) # (2*np.pi*Delta))
    DoAsESPRIT = np.radians(np.degrees(np.arcsin(-np.angle(eigs)/kappaDelta)) % 90) # (2*np.pi*Delta))
    # print('DoAs (\phi) sin = ', np.degrees(DoAsESPRIT))
    
    # DoAsESPRIT2 = np.radians(np.degrees(np.arcsin(-np.angle(eigs2)/kappaDelta)) % 90) 
    # print('DoAs (\phi2) sin = ', np.degrees(DoAsESPRIT2), '\n')
  
    return DoAsESPRIT
