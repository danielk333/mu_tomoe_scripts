import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import dpt

#import mu stuff
from mu_conf import *
from mu_gain import *

from scipy.integrate import quad


def sph_rng(min_el = 30.0):

    in_FOV = False
    while not in_FOV:
        on_sph = False
        while not on_sph:
            xi = np.random.rand(4)*2.0 - 1
            xin = xi.dot(xi)
            if xin < 1.0:
                on_sph = True

        x = 2.0*(xi[1]*xi[3] + xi[0]*xi[2])/xin
        y = 2.0*(xi[2]*xi[3] - xi[0]*xi[1])/xin
        z = (xi[0]**2 + xi[3]**2 - xi[1]**2 - xi[2]**2)/xin

        if np.arcsin(z) >= np.radians(min_el):
            in_FOV = True

    return np.array([x,y,-z], dtype=n.float)



def coll_area_samp(rcs_num, samp_n, max_range, snr_lim, height):

    rcs_dBsm = np.linspace(-50,20,num=rcs_num, dtype=np.float)

    size = np.sqrt(max_range**2 - height**2)

    samp = np.empty((3,samp_n), dtype=np.float)
    samp[2,:] = height

    ind = 0
    while ind < samp_n:
        samp[:2,ind] = (np.random.rand(2,)*2.0 - 1.0)*size
        if np.linalg.norm(samp[:2,ind]) < max_range:
            ind += 1
    
    samp_G = np.empty((samp_n,), dtype=np.float)

    for ind in range(samp_n):
        place = float(ind)/float(samp_n)*1000.0
        if np.abs(place - np.round(place)) < 1000.0*0.5/float(samp_n):
            print(f'{ind} of {samp_n} gain calc')
        samp_G[ind] = MU_gain(samp[:,ind])

    collection_area = np.zeros(rcs_dBsm.shape, dtype=rcs_dBsm.dtype)

    for rcsi, RCS in enumerate(rcs_dBsm):
        print(f'doing rcsi {rcsi}')
        for ind in range(samp_n):
            snr = MU_snr(10.0**((RCS+1.0)/10.0), samp[:,ind], G=samp_G[ind])
            if snr > snr_lim:
                collection_area[rcsi] += 1.0

    collection_area /= samp_n
    collection_area *= np.pi*(1e-3*size)**2

    return rcs_dBsm, collection_area





def coll_area_integ(rcs_num, min_el, snr_lim, height):

    rcs_dBsm = np.linspace(-50,20,num=rcs_num, dtype=np.float)

    def rv(r, theta):
        return np.array([r*np.cos(theta), r*np.sin(theta), height], dtype=np.float)

    collection_area = np.zeros(rcs_dBsm.shape, dtype=rcs_dBsm.dtype)

    def snr_fun(r, theta, RCS):
        if MU_snr(10.0**((RCS+1.0)/10.0), rv(r, theta)) > snr_lim:
            return r
        else:
            return 0.0

    def th_int(theta, RCS):
        return quad(snr_fun, 0, height/np.sin(np.radians(min_el)), args=(theta, RCS))[0]

    for rcsi, RCS in enumerate(rcs_dBsm):
        print(f'doing rcsi {rcsi}')
        collection_area[rcsi] = quad(th_int, 0.0, np.pi*2.0, args=(RCS,))[0]

    return rcs_dBsm, collection_area



if __name__ == '__main__':
    
    snr_lim = 1.0
    rcs_num = 200
    max_range = 127.0e3
    height = 100e3
    samp_n = 100000

    rcs_dBsm, collection_area = coll_area_samp(rcs_num, samp_n, max_range, snr_lim, height)

    fig = plt.figure(figsize=(14,12))

    ax = fig.add_subplot(111)
    ax.semilogy(rcs_dBsm, collection_area, '-b')
    ax.set_xlabel('RCS [dBsm]')
    ax.set_ylabel('Collection area [km$^2$]')

    fig.savefig('img/test_collection_area',bbox_inches='tight')
    plt.show()