import numpy as n
import scipy.constants as const

xpos = n.array([
 31.18, 27.28, 23.38, 19.49, 19.49, 15.59, 15.59, 11.69, 11.69,  7.79,  7.79,  3.90,  0.00,  0.00, -3.90, -3.90, -7.79, -7.79,-11.69,
 11.69, 11.69, 11.69,  7.79,  7.79,  7.79,  7.79,  3.90,  3.90,  3.90,  3.90,  3.90,  0.00,  0.00,  0.00,  0.00, -3.90, -3.90, -3.90,
 31.18, 31.18, 31.18, 27.28, 27.28, 27.28, 27.28, 23.38, 23.38, 23.38, 23.38, 23.38, 19.49, 19.49, 19.49, 19.49, 15.59, 15.59, 15.59,
 19.49, 19.49, 19.49, 15.59, 15.59, 15.59, 15.59, 11.69, 11.69, 11.69, 11.69, 11.69,  7.79,  7.79,  7.79,  7.79,  3.90,  3.90,  3.90,
 50.66, 50.66, 50.66, 50.66, 46.77, 46.77, 46.77, 46.77, 42.87, 42.87, 42.87, 42.87, 42.87, 38.97, 38.97, 38.97, 35.07, 35.07, 35.07,
 38.97, 38.97, 38.97, 35.07, 35.07, 35.07, 35.07, 31.18, 31.18, 31.18, 31.18, 31.18, 27.28, 27.28, 27.28, 27.28, 23.38, 23.38, 23.38,
 46.77, 46.77, 46.77, 42.87, 42.87, 42.87, 42.87, 38.97, 38.97, 38.97, 38.97, 38.97, 35.07, 35.07, 35.07, 35.07, 31.18, 31.18, 31.18,
 27.28, 27.28, 27.28, 23.38, 23.38, 23.38, 23.38, 19.49, 19.49, 19.49, 19.49, 19.49, 15.59, 15.59, 15.59, 15.59, 11.69, 11.69, 11.69,
 46.77, 46.77, 42.87, 42.87, 42.87, 38.97, 38.97, 38.97, 38.97, 35.07, 35.07, 31.18, 31.18, 31.18, 27.28, 27.28, 27.28, 23.38, 19.49,
 35.07, 35.07, 35.07, 31.18, 31.18, 31.18, 31.18, 27.28, 27.28, 27.28, 27.28, 27.28, 23.38, 23.38, 23.38, 23.38, 19.49, 19.49, 19.49,
 23.38, 23.38, 23.38, 19.49, 19.49, 19.49, 19.49, 15.59, 15.59, 15.59, 15.59, 15.59, 11.69, 11.69, 11.69, 11.69,  7.79,  7.79,  7.79,
 15.59, 15.59, 15.59, 11.69, 11.69, 11.69, 11.69,  7.79,  7.79,  7.79,  7.79,  7.79,  3.90,  3.90,  3.90,  3.90,  0.00,  0.00,  0.00,
 11.69,  7.79,  7.79,  3.90,  3.90,  0.00,  0.00, -3.90, -7.79, -7.79,-11.69,-11.69,-15.59,-15.59,-19.49,-19.49,-23.38,-27.28,-31.18,
  3.90,  3.90,  3.90,  0.00,  0.00,  0.00,  0.00, -3.90, -3.90, -3.90, -3.90, -3.90, -7.79, -7.79, -7.79, -7.79,-11.69,-11.69,-11.69,
-15.59,-15.59,-15.59,-19.49,-19.49,-19.49,-19.49,-23.38,-23.38,-23.38,-23.38,-23.38,-27.28,-27.28,-27.28,-27.28,-31.18,-31.18,-31.18,
 -3.90, -3.90, -3.90, -7.79, -7.79, -7.79, -7.79,-11.69,-11.69,-11.69,-11.69,-11.69,-15.59,-15.59,-15.59,-15.59,-19.49,-19.49,-19.49,
-35.07,-35.07,-35.07,-38.97,-38.97,-38.97,-42.87,-42.87,-42.87,-42.87,-42.87,-46.77,-46.77,-46.77,-46.77,-50.66,-50.66,-50.66,-50.66,
-23.38,-23.38,-23.38,-27.28,-27.28,-27.28,-27.28,-31.18,-31.18,-31.18,-31.18,-31.18,-35.07,-35.07,-35.07,-35.07,-38.97,-38.97,-38.97,
-31.18,-31.18,-31.18,-35.07,-35.07,-35.07,-35.07,-38.97,-38.97,-38.97,-38.97,-38.97,-42.87,-42.87,-42.87,-42.87,-46.77,-46.77,-46.77,
-11.69,-11.69,-11.69,-15.59,-15.59,-15.59,-15.59,-19.49,-19.49,-19.49,-19.49,-19.49,-23.38,-23.38,-23.38,-23.38,-27.28,-27.28,-27.28,
-19.49,-23.38,-27.28,-27.28,-27.28,-31.18,-31.18,-31.18,-35.07,-35.07,-38.97,-38.97,-38.97,-38.97,-42.87,-42.87,-42.87,-46.77,-46.77,
-19.49,-19.49,-19.49,-23.38,-23.38,-23.38,-23.38,-27.28,-27.28,-27.28,-27.28,-27.28,-31.18,-31.18,-31.18,-31.18,-35.07,-35.07,-35.07,
 -7.79, -7.79, -7.79,-11.69,-11.69,-11.69,-11.69,-15.59,-15.59,-15.59,-15.59,-15.59,-19.49,-19.49,-19.49,-19.49,-23.38,-23.38,-23.38,
  0.00,  0.00,  0.00, -3.90, -3.90, -3.90, -3.90, -7.79, -7.79, -7.79, -7.79, -7.79,-11.69,-11.69,-11.69,-11.69,-15.59,-15.59,-15.59,
  7.79,  7.79,  7.79,  3.90,  3.90,  3.90,  3.90,  0.00,  0.00,  0.00,  0.00,  0.00, -3.90, -3.90, -3.90, -3.90, -7.79, -7.79, -7.79
], dtype=n.float)

ypos = n.array([
 40.50, 42.75, 45.00, 47.25, 42.75, 45.00, 40.50, 47.25, 42.75, 49.50, 45.00, 47.25, 49.50, 45.00, 47.25, 42.75, 49.50, 45.00, 47.25,
 38.25, 33.75, 29.25, 40.50, 36.00, 31.50, 27.00, 42.75, 38.25, 33.75, 29.25, 24.75, 40.50, 36.00, 31.50, 27.00, 38.25, 33.75, 29.25,
 36.00, 31.50, 27.00, 38.25, 33.75, 29.25, 24.75, 40.50, 36.00, 31.50, 27.00, 22.50, 38.25, 33.75, 29.25, 24.75, 36.00, 31.50, 27.00,
 20.25, 15.75, 11.25, 22.50, 18.00, 13.50,  9.00, 24.75, 20.25, 15.75, 11.25,  6.75, 22.50, 18.00, 13.50,  9.00, 20.25, 15.75, 11.25,
  6.75,  2.25, -2.25, -6.75, 18.00, 13.50,  9.00,  4.50, 24.75, 20.25, 15.75, 11.25,  6.75, 31.50, 27.00, 22.50, 33.75, 29.25, 24.75,
 18.00, 13.50,  9.00, 20.25, 15.75, 11.25,  6.75, 22.50, 18.00, 13.50,  9.00,  4.50, 20.25, 15.75, 11.25,  6.75, 18.00, 13.50,  9.00,
  0.00, -4.50, -9.00,  2.25, -2.25, -6.75,-11.25,  4.50,  0.00, -4.50, -9.00,-13.50,  2.25, -2.25, -6.75,-11.25,  0.00, -4.50, -9.00,
  2.25, -2.25, -6.75,  4.50,  0.00, -4.50, -9.00,  6.75,  2.25, -2.25, -6.75,-11.25,  4.50,  0.00, -4.50, -9.00,  2.25, -2.25, -6.75,
-13.50,-18.00,-15.75,-20.25,-24.75,-18.00,-22.50,-27.00,-31.50,-29.25,-33.75,-31.50,-36.00,-40.50,-33.75,-38.25,-42.75,-45.00,-47.25,
-15.75,-20.25,-24.75,-13.50,-18.00,-22.50,-27.00,-11.25,-15.75,-20.25,-24.75,-29.25,-13.50,-18.00,-22.50,-27.00,-15.75,-20.25,-24.75,
-31.50,-36.00,-40.50,-29.25,-33.75,-38.25,-42.75,-27.00,-31.50,-36.00,-40.50,-45.00,-29.25,-33.75,-38.25,-42.75,-31.50,-36.00,-40.50,
-13.50,-18.00,-22.50,-11.25,-15.75,-20.25,-24.75, -9.00,-13.50,-18.00,-22.50,-27.00,-11.25,-15.75,-20.25,-24.75,-13.50,-18.00,-22.50,
-47.25,-45.00,-49.50,-42.75,-47.25,-45.00,-49.50,-47.25,-45.00,-49.50,-42.75,-47.25,-40.50,-45.00,-42.75,-47.25,-45.00,-42.75,-40.50,
-29.25,-33.75,-38.25,-27.00,-31.50,-36.00,-40.50,-24.75,-29.25,-33.75,-38.25,-42.75,-27.00,-31.50,-36.00,-40.50,-29.25,-33.75,-38.25,
-27.00,-31.50,-36.00,-24.75,-29.25,-33.75,-38.25,-22.50,-27.00,-31.50,-36.00,-40.50,-24.75,-29.25,-33.75,-38.25,-27.00,-31.50,-36.00,
-11.25,-15.75,-20.25, -9.00,-13.50,-18.00,-22.50, -6.75,-11.25,-15.75,-20.25,-24.75, -9.00,-13.50,-18.00,-22.50,-11.25,-15.75,-20.25,
-24.75,-29.25,-33.75,-22.50,-27.00,-31.50, -6.75,-11.25,-15.75,-20.25,-24.75, -4.50, -9.00,-13.50,-18.00,  6.75,  2.25, -2.25, -6.75,
 -9.00,-13.50,-18.00, -6.75,-11.25,-15.75,-20.25, -4.50, -9.00,-13.50,-18.00,-22.50, -6.75,-11.25,-15.75,-20.25, -9.00,-13.50,-18.00,
  9.00,  4.50,  0.00, 11.25,  6.75,  2.25, -2.25, 13.50,  9.00,  4.50,  0.00, -4.50, 11.25,  6.75,  2.25, -2.25,  9.00,  4.50,  0.00,
  6.75,  2.25, -2.25,  9.00,  4.50,  0.00, -4.50, 11.25,  6.75,  2.25, -2.25, -6.75,  9.00,  4.50,  0.00, -4.50,  6.75,  2.25, -2.25,
 47.25, 45.00, 42.75, 38.25, 33.75, 40.50, 36.00, 31.50, 33.75, 29.25, 31.50, 27.00, 22.50, 18.00, 24.75, 20.25, 15.75, 18.00, 13.50,
 24.75, 20.25, 15.75, 27.00, 22.50, 18.00, 13.50, 29.25, 24.75, 20.25, 15.75, 11.25, 27.00, 22.50, 18.00, 13.50, 24.75, 20.25, 15.75,
 40.50, 36.00, 31.50, 42.75, 38.25, 33.75, 29.25, 45.00, 40.50, 36.00, 31.50, 27.00, 42.75, 38.25, 33.75, 29.25, 40.50, 36.00, 31.50,
 22.50, 18.00, 13.50, 24.75, 20.25, 15.75, 11.25, 27.00, 22.50, 18.00, 13.50,  9.00, 24.75, 20.25, 15.75, 11.25, 22.50, 18.00, 13.50,
  4.50,  0.00, -4.50,  6.75,  2.25, -2.25, -6.75,  9.00,  4.50,  0.00, -4.50, -9.00,  6.75,  2.25, -2.25, -6.75,  4.50,  0.00, -4.50
  ], dtype=n.float)

zpos = n.zeros(ypos.shape, dtype = ypos.dtype)
xpos.shape = (xpos.size,)
ypos.shape = (ypos.size,)
zpos.shape = (zpos.size,)

MU_antennas = n.empty((n.prod(ypos.shape), 3), dtype=n.float)
for ind in range(n.prod(ypos.shape)):
    MU_antennas[ind,0] = xpos[ind]
    MU_antennas[ind,1] = ypos[ind]
    MU_antennas[ind,2] = zpos[ind]

f0 = 46.5e6
lambda0=const.c/f0;
gain_yagi = 7.24
I_0 = gain_yagi*xpos.size
bw0 = 1.0/6e-6
tx_P = 1.0e6

kiso_lat = (35.0 + (47.0 + 50.28/60.0)/60.0 );
kiso_lon = (137.0 + (37.0 + 31.26/60.0)/60.0 );
kiso_alt = 1168;

mu_lat = (34.0 + (51.0 + 14.50/60.0)/60.0 );
mu_lon = (136.0 + (06.0 + 20.24/60.0)/60.0 );
mu_alt = 372.0#