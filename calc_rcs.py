import numpy as n
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import dpt

#import mu stuff
from mu_conf import *
from mu_gain import *


if __name__ == '__main__':

    tomoe_part = n.genfromtxt('data/tomoe_part.csv')

    #FOR DEBUG, JUST DO ONE
    tomoe_part = tomoe_part[:1,:]

    cart_shift = n.array([0,0,100e3],dtype=n.float)

    az0, el0, r0 = dpt.convert.cart_to_azel(cart_shift[0], cart_shift[1], cart_shift[2], radians=False)
    start = dpt.convert.azel_to_azel_via_ecef(
    mu_lat,mu_lon,mu_alt,
    az0,el0,r0,
    kiso_lat,kiso_lon,kiso_alt)
    dir_kiso = dpt.convert.azel_to_cart(n.radians(start[0]), n.radians(start[1]), start[2])

    plot_beam = True
    res = 100
    rho = n.linspace(0, 1, endpoint=True, num=res, dtype=n.float)

    for j in range(tomoe_part.shape[0]):
        cart0t = dpt.convert.azel_to_cart(n.radians(tomoe_part[j, 0]), n.radians(tomoe_part[j, 1]), tomoe_part[j, 2])
        cart1t = dpt.convert.azel_to_cart(n.radians(tomoe_part[j, 3]), n.radians(tomoe_part[j, 4]), tomoe_part[j, 5])

        cart0tn = cart0t/n.linalg.norm(cart0t)
        cart0tn[2] += 10.0*n.log10(I_0) + 3.0
        cart1tn = cart1t/n.linalg.norm(cart1t)
        cart1tn[2] += 10.0*n.log10(I_0) + 3.0

        gain_curve = n.empty(rho.shape,dtype=n.float)
        snr_curve = n.full(rho.shape, 10.0, dtype=n.float)
        rcs_curve = n.empty(rho.shape,dtype=n.float)
        for ind in range(res):
            r = cart0t*(1.0 - rho[ind]) + cart1t*rho[ind]
            gain_curve[ind] = MU_gain(r)
            rcs_curve[ind] = MU_rcs(snr_curve[ind], r)

        if plot_beam:
            fig = plt.figure(figsize=(14,12))

            ax = fig.add_subplot(231)
            ax.plot(rho, 10.0*n.log10(gain_curve))
            ax.set_xlabel('Trajectory')
            ax.set_ylabel('Gain [dB]')

            ax = fig.add_subplot(232)
            ax.plot(rho, 10.0*n.log10(snr_curve))
            ax.set_xlabel('Trajectory')
            ax.set_ylabel('SNR [dB]')

            ax = fig.add_subplot(233)
            ax.plot(rho, 10.0*n.log10(rcs_curve*100.0**2))
            ax.set_xlabel('Trajectory')
            ax.set_ylabel('RCS [dB$cm^2$]')

            ax = fig.add_subplot(212, projection='3d')
            add_ax_gain3d(ax, 50, 80.0)
            ax.plot([cart0tn[0], cart1tn[0]], [cart0tn[1], cart1tn[1]], [cart0tn[2], cart1tn[2]], '.-r')
            ax.set_xlabel('$k_x$ [1]')
            ax.set_ylabel('$k_y$ [1]')
            ax.set_zlabel('Gain $G$ [dB]')

            fig.savefig('img/test_rcs_curve_calc',bbox_inches='tight')
            plt.show()


