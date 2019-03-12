import numpy as n
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from mu_conf import *

def plane_wave(k,r,p):
    return n.exp(1j*n.pi*2.0*n.dot(k-p,r))

def array(k, antennas, f, I_0):
    I_scale = I_0/(antennas.shape[0]**2.0)
    k = k/n.linalg.norm(k)
    p = n.array([0,0,1], dtype=k.dtype)
    G = n.exp(1j)*0.0
    #r in meters, divide by lambda
    for r in antennas:
        G += plane_wave(k,r/(const.c/f),p)
    #ugly fix, multiply gain by k_z to enmulate beam stering loss as a function of elevation
    #should be antenna element gain pattern of k...
    return n.abs(G.conj()*G*I_scale)*p[2]


def MU_gain(k):
    return array(k, MU_antennas, f0, I_0)

def add_ax_gain3d(ax, res, min_el):
    kx=n.linspace(-n.cos(min_el*n.pi/180.0),n.cos(min_el*n.pi/180.0),num=res)
    ky=n.linspace(-n.cos(min_el*n.pi/180.0),n.cos(min_el*n.pi/180.0),num=res)
    
    S=n.zeros((res,res))
    K=n.zeros((res,res,2))
    for i,x in enumerate(kx):
        for j,y in enumerate(ky):
            z2 = x**2 + y**2
            if z2 < n.cos(min_el*n.pi/180.0)**2:
                k = n.array([x,y,n.sqrt(1.0-z2)], dtype=n.float)
                S[i,j]=MU_gain(k)
            else:
                S[i,j] = 0;
            K[i,j,0]=x
            K[i,j,1]=y
    SdB = n.log10(S)*10.0
    SdB[SdB < -5.0] = -5.0
    surf = ax.plot_surface(K[:,:,0],K[:,:,1],SdB,cmap=cm.plasma, linewidth=0, antialiased=False, vmin=0, vmax=n.max(SdB))


def plot_gain3d(res=200,min_el = 0.0):
    plt.rc('text', usetex=True)
    
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111, projection='3d')
    
    add_ax_gain3d(ax, res, min_el)
    ax.set_xlabel('$k_x$ [1]',fontsize=24)
    ax.set_ylabel('$k_y$ [1]',fontsize=24)
    ax.set_zlabel('Gain $G$ [dB]',fontsize=24)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    ax.set_title('Gain pattern MU', fontsize=28)
    plt.show()

def SNR_to_RCS(SNR, R, rx_T, G_tx, G_rx, wavelength, rx_bw, tx_power):
    c1 = (4.0*n.pi)**3*R**4*const.k*rx_T*rx_bw
    c2 = tx_power*G_tx*G_rx*wavelength**2
    return SNR*c1/c2

def RCS_to_SNR(RCS, R, rx_T, G_tx, G_rx, wavelength, rx_bw, tx_power):
    c1 = (4.0*n.pi)**3*R**4*const.k*rx_T*rx_bw
    c2 = tx_power*G_tx*G_rx*wavelength**2
    return RCS*c2/c1

def MU_rcs(SNR, r, T_cos = 10000.0, G = None):
    if G is None:
        G = MU_gain(r)
    rx_T = 3000.0 + T_cos
    return SNR_to_RCS(SNR, n.linalg.norm(r), rx_T, G, G, lambda0, bw0, tx_P)

def MU_snr(RCS, r, T_cos = 10000.0, G = None):
    if G is None:
        G = MU_gain(r)
    rx_T = 3000.0 + T_cos
    return RCS_to_SNR(RCS, n.linalg.norm(r), rx_T, G, G, lambda0, bw0, tx_P)


if __name__=='__main__':
    pass

