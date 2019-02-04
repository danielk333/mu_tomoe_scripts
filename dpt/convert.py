"""
.. module:: convert-tools
   :platform: Unix, Windows
   :synopsis: This is a module to collect verified conversion functions, such as between differnt times, coordinate systems, units ect

.. moduleauthor:: Daniel Kastinen <daniel.kastinen@irf.se>


Copied from DPT repository hash=2309d8c

CHANGELOG from DPT state:

  30-01-2019: removed functions not used

"""

#import astropy.units as asu
#from astropy.coordinates import SkyCoord

import pyproj
import numpy as np
import scipy.constants
from numpy import cos,sin,tan,arctan2,sqrt,arctan
import matplotlib.pyplot as plt

import pdb


def gps_to_ecef_pyproj(lat=None, lon=None, alt=None, lla=None, radians=False):
	"""Convert from GPS coordinates in WGS84 to ECEF coordinates using the pyproj library

	Args:
		:lat (rads)  :  latitude, north +, south -, range [-pi/2,pi/2], equator = 0.
		:lon (rads)  :  longitude, east +, west -, range [-pi,pi], prime meridian 0.
		:alt (meters):  Altitude above geode in meters.
		:lla (vector):  Contains lat lon alt as a vector
	Example::
		
		kiso_lat = 35.794167*np.pi/180.0;
		kiso_lon = 137.628333*np.pi/180.0;
		kiso_alt = 1130;

		kiso_ecef = gps_to_ecef_pyproj(kiso_lat,kiso_lon,kiso_alt)
		...


	Return:
		:r (3d vector meters): numpy array vector of [x,y,z] coordinates in ECEF

	No Raises
	"""
	if alt is None:
		alt = 0.
	if (lat is None or lon is None) and lla is None:
		raise ValueError('Need lat and lon OR lla: lat={},lon={},alt={},lla={}'\
			.format(lat,lon,alt,lla))

	if lla is not None:
		lat = lla[0]
		lon = lla[1]
		alt = lla[2]

	ecef_p = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
	lla_p = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
	x, y, z = pyproj.transform(lla_p, ecef_p, lon, lat, alt, radians=radians)

	return np.array([x, y, z])

def ecef_to_gps_pyproj(x=None, y=None, z=None, ecef=None, radians=False):
	"""Convert from ECEF coordinates to GPS coordinates in WGS84 using the pyproj library

	"""
	if (x is None or y is None) and ecef is None:
		raise ValueError('Need x,y and z OR ecef: x={},y={},z={},ecef={}'\
			.format(x,y,z,ecef))

	if ecef is not None:
		x = ecef[0]
		y = ecef[1]
		z = ecef[2]
	ecef_p = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
	lla_p = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
	lon, lat, alt = pyproj.transform(ecef_p, lla_p, x, y, z, radians=radians)

	return np.array([lat, lon, alt])

def azel_to_cart(az,el,r):
	"""Convert from spherical coordinates to cartesian in a degrees east of north and elevation fashion

	"""
	return r*np.array([sin(az)*cos(el), cos(az)*cos(el), sin(el)])

def cart_to_azel(x,y,z,radians=True):
	"""Convert from cartesian coordinates to spherical in a degrees east of north and elevation fashion

	"""
	ans = np.array([ np.pi/2 - arctan2(y,x), arctan(z/(sqrt(x**2 + y**2))),sqrt(x**2 + y**2 +z**2) ])
	if not radians:
		ans[:2] = np.degrees(ans[:2])
	return ans

def rot_mat(a,b):
	"""Create matrix to rotate vector a to vector b
		
	Reference: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677
	"""

	a = a/np.linalg.norm(a)
	b = b/np.linalg.norm(b)

	v = np.cross(a,b)
	c = np.dot(a,b)

	Vx = np.array([ \
		[0	,-v[2], v[1]], \
		[ v[2],0	,-v[0]], \
		[-v[1], v[0],0	]])

	R = np.eye(3,3) + Vx + np.matmul(Vx,Vx)*(1.0/(1.0+c))

	return R


def rot_mat_z(theta):
	R = np.zeros((3,3))
	R[0,0] = np.cos(theta)
	R[0,1] = -np.sin(theta)
	R[1,0] = np.sin(theta)
	R[1,1] = np.cos(theta)
	R[2,2] = 1.0
	return R

def rot_mat_x(theta):
	R = np.zeros((3,3))
	R[1,1] = np.cos(theta)
	R[1,2] = -np.sin(theta)
	R[2,1] = np.sin(theta)
	R[2,2] = np.cos(theta)
	R[0,0] = 1.0
	return R

def rot_mat_y(theta):
	R = np.zeros((3,3))
	R[0,0] = np.cos(theta)
	R[0,2] = np.sin(theta)
	R[2,0] = -np.sin(theta)
	R[2,2] = np.cos(theta)
	R[1,1] = 1.0
	return R


def azel_to_azel_via_ecef(lat0,lon0,alt0,az,el,r,lat1,lon1,alt1, radians=False):
	"""Transfer the spherical coordinates for a point relative location 0 on the earth to spherical coordinates relative location 1
	
	"""

	if not radians:
		lat0=np.radians(lat0)
		lon0=np.radians(lon0)
		az=np.radians(az)
		el=np.radians(el)
		lat1=np.radians(lat1)
		lon1=np.radians(lon1)

	zenith = np.array([0,0,1.0])
	north = np.array([0,1.0,0])
	east = np.array([1.0,0,0])

	p_0_ref = azel_to_cart(az,el,r)
	ecef_0 = gps_to_ecef_pyproj(lat0,lon0,alt0,radians=True)
	ecef_1 = gps_to_ecef_pyproj(lat1,lon1,alt1,radians=True)

	north1 = -1.0*ecef_1
	north1[2] = 0.0; #xy plane projection
	north1 = north1/np.linalg.norm(north1)
	north1_rotation = rot_mat(north1,north)

	north0 = -1.0*ecef_0
	north0[2] = 0.0; #xy plane projection
	north0 = north0/np.linalg.norm(north0)
	north0_rotation = rot_mat(north,north0)

	R_0_zenith_to_ecef = rot_mat(zenith,ecef_0)
	p_ecef_0_ref = north0_rotation.dot(p_0_ref)
	p_ecef_0_ref = R_0_zenith_to_ecef.dot(p_ecef_0_ref)

	p_ecef = ecef_0 + p_ecef_0_ref;
	p_ecef_1_ref = p_ecef - ecef_1;


	R_1_ecef_to_zenith = rot_mat(ecef_1,zenith)

	p_1_ref = R_1_ecef_to_zenith.dot(p_ecef_1_ref)
	p_1_ref = north1_rotation.dot(p_1_ref)

	azel_end = cart_to_azel(p_1_ref[0],p_1_ref[1],p_1_ref[2])

	if not radians:
		azel_end[:2] = np.degrees(azel_end[:2])

	return azel_end

def np_vec_norm(x,axis):
	return np.sqrt(np.sum(x**2,axis=axis))

