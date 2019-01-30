import sys
import os

import numpy as n
import dpt
import matplotlib.pyplot as plt
fout = 'data/MU_det_KISO_azel.txt'

if os.path.isfile(fout):
	KISO_MU = n.genfromtxt(fout,delimiter=',')
	
	fig, ax = plt.subplots(figsize=(15, 7))
	ax.plot(KISO_MU[:,6],KISO_MU[:,7],'.b')
	ax.set(xlabel='az [deg]',ylabel='el [deg]',title='MU detections (start point) in KISO reference frame')
	fig.savefig('img/mu_det_kiso_frame',bbox_inches='tight')

	fig, ax = plt.subplots(figsize=(15, 7))
	for row in KISO_MU:
		ax.plot([row[6],row[9]],[row[7],row[10]],'-b',alpha=0.2)
	ax.set(xlabel='az [deg]',ylabel='el [deg]',title='MU detections in KISO reference frame')
	fig.savefig('img/mu_traj_kiso_frame',bbox_inches='tight')

	plt.show()

else:

	XYZ_MU = n.genfromtxt('data/MUinput_xyz_v8_beam_1804.txt',delimiter=',')
	M = XYZ_MU.shape[0]

	XYZ_MU[:,6:]*=1e3

	kiso_lat = (35.0 + (47.0 + 50.28/60.0)/60.0 );
	kiso_lon = (137.0 + (37.0 + 31.26/60.0)/60.0 );
	kiso_alt = 1168;

	#mu_lat = 34.8540#
	#mu_lon = 136.1044# MU radar
	#mu_alt = 380.0#
	mu_lat = (34.0 + (51.0 + 14.50/60.0)/60.0 );
	mu_lon = (136.0 + (06.0 + 20.24/60.0)/60.0 );
	mu_alt = 372.0#

	KISO_MU = n.empty( (M,12) )
	KISO_MU[:,:6] = XYZ_MU[:,:6].copy()

	for i in range(M):

		az0,el0,r0 = dpt.convert.cart_to_azel(XYZ_MU[i,6],XYZ_MU[i,7],XYZ_MU[i,8],radians=False)
		#print('START:',az0,el0,r0)
		start = dpt.convert.azel_to_azel_via_ecef(\
			mu_lat,mu_lon,mu_alt,\
			az0,el0,r0,\
			kiso_lat,kiso_lon,kiso_alt)

		az0,el0,r0 = dpt.convert.cart_to_azel(XYZ_MU[i,10],XYZ_MU[i,11],XYZ_MU[i,12],radians=False)
		#print('END:',az0,el0,r0)
		end   = dpt.convert.azel_to_azel_via_ecef(\
			mu_lat,mu_lon,mu_alt,\
			az0,el0,r0,\
			kiso_lat,kiso_lon,kiso_alt)

		KISO_MU[i,6:9] = start
		KISO_MU[i,9:] = end


	n.savetxt(fout,KISO_MU,delimiter=',')

'''
% xyz data:
%    1. Year
%    2. Month
%    3. [UT] Day+fraction of day (JST is UT+8 hours  -> we subtract 9 hours)
%    4. [JST] Day
%    5. [JST] Hour, minute, second
%    6. [JST] Microsecond (.ssssss)
%    7. [km] Start point Azimuth (degrees east of north)
%    8. [km] Start point Elevation (degrees)
%    9. [km] Start point Range (meters)
%    10.[km] End point Azimuth (degrees east of north)
%    11.[km] End point Elevation (degrees)
%    12.[km] End point Range (meters)
'''
