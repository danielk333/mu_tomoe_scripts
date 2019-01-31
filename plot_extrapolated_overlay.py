import sys

import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dpt

tomoe_part = n.genfromtxt('data/tomoe_part.csv')

dat = n.genfromtxt('data/kiso.csv', delimiter=',', skip_header = 1)
dat_mu = n.genfromtxt('data/MU_orbit_data_20181023.txt', delimiter=' ')

plot_n = [3]

if 1 in plot_n:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(tomoe_part.shape[0]):
        met_id = dat[j, 0]
        mu_id = n.argmin(n.abs(dat_mu[:, 0] - met_id))
        
        cart0t = dpt.convert.azel_to_cart(n.radians(tomoe_part[j, 0]), n.radians(tomoe_part[j, 1]), tomoe_part[j, 2])
        cart1t = dpt.convert.azel_to_cart(n.radians(tomoe_part[j, 3]), n.radians(tomoe_part[j, 4]), tomoe_part[j, 5])
        
        az0 = n.radians(dat_mu[mu_id, 22])
        el0 = n.radians(90.0 - dat_mu[mu_id, 23])
        r0 = dat_mu[mu_id, 20]/n.sin(el0)*1e3
        cart0 = dpt.convert.azel_to_cart(az0, el0, r0)
        az1 = n.radians(dat_mu[mu_id, 24])
        el1 = n.radians(90.0 - dat_mu[mu_id, 25])
        r1 = dat_mu[mu_id, 21]/n.sin(el1)*1e3
        cart1 = dpt.convert.azel_to_cart(az1, el1, r1)
        
        ax.plot([cart0[0], cart1[0]], [cart0[1], cart1[1]], [cart0[2], cart1[2]], '-k')
        ax.plot([cart0t[0], cart1t[0]], [cart0t[1], cart1t[1]], [cart0t[2], cart1t[2]], '.r')

    fig.savefig('img/overlay_3d_plot',bbox_inches='tight')
        

if 2 in plot_n:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(tomoe_part.shape[0]):
        cart0t = dpt.convert.azel_to_cart(n.radians(tomoe_part[j, 0]), n.radians(tomoe_part[j, 1]), tomoe_part[j, 2])
        cart1t = dpt.convert.azel_to_cart(n.radians(tomoe_part[j, 3]), n.radians(tomoe_part[j, 4]), tomoe_part[j, 5])
        
        ax.plot([cart0t[0], cart1t[0]], [cart0t[1], cart1t[1]], [cart0t[2], cart1t[2]], '-k', alpha=0.5)
        
    fig.savefig('img/kiso_at_MU_3d',bbox_inches='tight')

if 3 in plot_n:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(0.5*(tomoe_part[:,8] + tomoe_part[:,7]))
    ax.set_xlabel('midpoint fraction tomoe of mu')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(tomoe_part[:,8] - tomoe_part[:,7])
    ax.set_xlabel('reach fraction tomoe of mu')

    fig.savefig('img/overlay_statistics',bbox_inches='tight')

plt.show()
