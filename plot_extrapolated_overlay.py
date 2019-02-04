import sys

import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dpt

tomoe_part = n.genfromtxt('data/tomoe_part.csv')

dat = n.genfromtxt('data/kiso.csv', delimiter=',', skip_header = 1)
dat_mu = n.genfromtxt('data/MU_orbit_data_20181023.txt', delimiter=' ')

plot_n = [2,4]

cart_shift = n.array([0,0,100e3],dtype=n.float)
kiso_lat = (35.0 + (47.0 + 50.28/60.0)/60.0 );
kiso_lon = (137.0 + (37.0 + 31.26/60.0)/60.0 );
kiso_alt = 1168;

mu_lat = (34.0 + (51.0 + 14.50/60.0)/60.0 );
mu_lon = (136.0 + (06.0 + 20.24/60.0)/60.0 );
mu_alt = 372.0#

az0, el0, r0 = dpt.convert.cart_to_azel(cart_shift[0], cart_shift[1], cart_shift[2], radians=False)
start = dpt.convert.azel_to_azel_via_ecef(
    mu_lat,mu_lon,mu_alt,
    az0,el0,r0,
    kiso_lat,kiso_lon,kiso_alt)
dir_kiso = dpt.convert.azel_to_cart(n.radians(start[0]), n.radians(start[1]), start[2])

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

        cart0t = (cart0t - cart_shift)*1e-3
        cart1t = (cart1t - cart_shift)*1e-3
        
        ax.plot([cart0t[0], cart1t[0]], [cart0t[1], cart1t[1]], [cart0t[2], cart1t[2]], '-k', alpha=0.5)

    dir_kison = dir_kiso/n.linalg.norm(dir_kiso)

    ax.plot([0.0, dir_kison[0]*10.0], [0.0, dir_kison[1]*10.0], [0.0, dir_kison[2]*10.0], '-r')
    ax.set_aspect('equal')
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('Zenith')
    fig.savefig('img/kiso_at_MU_3d',bbox_inches='tight')

if 4 in plot_n:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #The reason we do not just use rot_mat here is that when calculating the 
    #matrix that transforms the normal of the image plane to align with the north direction
    #there is an ambiguity in the final rotation AROUND the north direction, 
    #this rot_mat takes the shortest route on the sphere to create the
    #transformation while a "az-el" composite matrix rotations makes sure that
    #the horizon (composite of east and north in old system) is still parallel to x-axis in new system
    #rot_kiso = dpt.convert.rot_mat(dir_kiso, n.array([0,1,0],dtype=n.float))

    rot_mat_x = dpt.convert.rot_mat_x(-n.radians(start[1]))
    rot_mat_z = dpt.convert.rot_mat_z(n.radians(start[0]))

    rot_kiso = rot_mat_x.dot(rot_mat_z)

    for j in range(tomoe_part.shape[0]):
        cart0t = dpt.convert.azel_to_cart(n.radians(tomoe_part[j, 0]), n.radians(tomoe_part[j, 1]), tomoe_part[j, 2])
        cart1t = dpt.convert.azel_to_cart(n.radians(tomoe_part[j, 3]), n.radians(tomoe_part[j, 4]), tomoe_part[j, 5])

        cart0t = rot_kiso.dot(cart0t - cart_shift)*1e-3
        cart1t = rot_kiso.dot(cart1t - cart_shift)*1e-3
        
        ax.plot([cart0t[0], cart1t[0]], [cart0t[2], cart1t[2]], '-k', alpha=0.5)

    ax.set_aspect('equal')
    fig.savefig('img/kiso_at_MU_proj',bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(tomoe_part.shape[0]):
        cart0t = dpt.convert.azel_to_cart(n.radians(tomoe_part[j, 0]), n.radians(tomoe_part[j, 1]), tomoe_part[j, 2])
        cart1t = dpt.convert.azel_to_cart(n.radians(tomoe_part[j, 3]), n.radians(tomoe_part[j, 4]), tomoe_part[j, 5])

        cart0t = (cart0t - cart_shift)*1e-3
        cart1t = (cart1t - cart_shift)*1e-3

        cart0t = rot_kiso.dot(cart0t)
        cart1t = rot_kiso.dot(cart1t)

        ax.plot([cart0t[0], cart1t[0]], [cart0t[1], cart1t[1]], [cart0t[2], cart1t[2]], '-k', alpha=0.5)

    dir_kison = dir_kiso/n.linalg.norm(dir_kiso)
    dir_kison = rot_kiso.dot(dir_kison)

    ax.plot([0.0, dir_kison[0]*10.0], [0.0, dir_kison[1]*10.0], [0.0, dir_kison[2]*10.0], '-r')
    ax.set_aspect('equal')
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('Zenith')
    fig.savefig('img/kiso_at_MU_proj_3d',bbox_inches='tight')

if 3 in plot_n:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(0.5*(tomoe_part[:,8] + tomoe_part[:,7]))
    ax.set_xlabel('midpoint fraction tomoe of mu')

    fig.savefig('img/overlay_statistics1',bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(tomoe_part[:,8] - tomoe_part[:,7])
    ax.set_xlabel('reach fraction tomoe of mu')

    fig.savefig('img/overlay_statistics2',bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.hist(tomoe_part[:,7])
    ax.set_xlabel('start point tomoe in mu')
    ax = fig.add_subplot(212)
    ax.hist(tomoe_part[:,8])
    ax.set_xlabel('end point tomoe in mu')

    fig.savefig('img/overlay_statistics3',bbox_inches='tight')

plt.show()
