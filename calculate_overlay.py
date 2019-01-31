import sys

import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dpt

dat = n.genfromtxt('data/kiso.csv', delimiter=',', skip_header = 1)
dat_mu = n.genfromtxt('data/MU_orbit_data_20181023.txt', delimiter=' ')
dat_kiso = n.genfromtxt('data/MUdata_20181023.txt', delimiter=',')

print(dat_kiso.shape)
print(dat.shape)
print(dat_mu.shape)

# Kiso observatory
latK = (35.0 + (47.0 + 50.28/60.0)/60.0 );
lonK = (137.0 + (37.0 + 31.26/60.0)/60.0 );
hgtK = 1168;

# MU radar
latS = (34.0 + (51.0 + 14.50/60.0)/60.0 );
lonS = (136.0 + (06.0 + 20.24/60.0)/60.0 );
hgtS = 372.0#

dat_conv = n.empty((dat.shape[0], 4))

res = 200

verb = False

tomoe_part = n.empty((dat.shape[0], 9), dtype=n.float)

#save rho to find rho distribution?

for i in range(dat.shape[0]):
    met_id = dat[i, 0]
    mu_id = n.argmin(n.abs(dat_mu[:, 0] - met_id))
    
    #start
    az0_kiso = dat[i, 2]
    el0_kiso = dat[i, 3]
    #end
    az1_kiso = dat[i, 4]
    el1_kiso = dat[i, 5]
    
    az0_tomoe = dat[i, 7]
    el0_tomoe = dat[i, 8]
    #end
    az1_tomoe = dat[i, 9]
    el1_tomoe = dat[i, 10]
    
    if verb:
        print('kiso:',az0_kiso, el0_kiso, az1_kiso, el1_kiso)
    
    az0 = n.radians(dat_mu[mu_id, 22])
    el0 = n.radians(90.0 - dat_mu[mu_id, 23])
    r0 = dat_mu[mu_id, 20]/n.sin(el0)*1e3
    cart0 = dpt.convert.azel_to_cart(az0, el0, r0)
    az1 = n.radians(dat_mu[mu_id, 24])
    el1 = n.radians(90.0 - dat_mu[mu_id, 25])
    r1 = dat_mu[mu_id, 21]/n.sin(el1)*1e3
    cart1 = dpt.convert.azel_to_cart(az1, el1, r1)
    
    if verb:
        print('start sph : ',az0,el0,r0)
        print('start cart: ',cart0)
        print('stop sph  : ',az1,el1,r1)
        print('stop cart : ', cart1)
        
    mu_set_kiso = n.empty((3,res))
    mu_set_kiso_start_d = n.empty((res,))
    mu_set_kiso_end_d = n.empty((res,))
    rho = n.linspace(0, 1, num=res, endpoint=True, dtype=n.float)
    for j in range(res):
        cart_r = cart0*(1.0 - rho[j]) + cart1*rho[j]
        mu_set = dpt.convert.cart_to_azel(cart_r[0], cart_r[1], cart_r[2])
        mu_set[:2] = n.degrees(mu_set[:2])
        tmp_set = dpt.convert.azel_to_azel_via_ecef(latS,lonS,hgtS,mu_set[0],mu_set[1],mu_set[2],latK,lonK,hgtK)
        
        mu_set_kiso_start_d[j] = n.linalg.norm(tmp_set[:2] - dat[i, 7:9])
        mu_set_kiso_end_d[j] = n.linalg.norm(tmp_set[:2] - dat[i, 9:11])
        
        mu_set_kiso[:,j] = tmp_set
    
    start_ind = n.argmin(mu_set_kiso_start_d)
    end_ind = n.argmin(mu_set_kiso_end_d)
    
    tomoe_start_mu = mu_set_kiso[:,start_ind]
    tomoe_end_mu = mu_set_kiso[:,end_ind]
    
    tomoe_part[i, :3] = dpt.convert.azel_to_azel_via_ecef(latK,lonK,hgtK,tomoe_start_mu[0],tomoe_start_mu[1],tomoe_start_mu[2],latS,lonS,hgtS)
    tomoe_part[i, 3:6] = dpt.convert.azel_to_azel_via_ecef(latK,lonK,hgtK,tomoe_end_mu[0],tomoe_end_mu[1],tomoe_end_mu[2],latS,lonS,hgtS)
    tomoe_part[i, 6] = dat_mu[mu_id, 0]
    tomoe_part[i, 7] = rho[start_ind]
    tomoe_part[i, 8] = rho[end_ind]
        
    if verb:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mu_set_kiso[0,:], mu_set_kiso[1,:], '.k', label='mu traj')
        ax.plot([tomoe_start_mu[0], tomoe_end_mu[0]], [tomoe_start_mu[1], tomoe_end_mu[1]], 'xr', label='mu traj kiso part')
        ax.plot([az0_kiso, az1_kiso], [el0_kiso, el1_kiso], 'or', label='kiso az-el')
        ax.plot([az0_tomoe, az1_tomoe], [el0_tomoe, el1_tomoe], 'ob', label='tomoe az-el')
        ax.plot([dat_kiso[mu_id,14], dat_kiso[mu_id,17]], [dat_kiso[mu_id,15], dat_kiso[mu_id,18]], 'xm', label='tomoe az-el send')
        
        
        plt.legend()
        plt.show()
    
    print('{} of {} done'.format(i+1,dat.shape[0]))


n.savetxt('data/tomoe_part.csv', tomoe_part)
