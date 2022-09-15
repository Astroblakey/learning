import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.integrate import odeint
from numpy import sqrt
import math
from math import *
import sys
# This code is for Blake to test out random variables and form a covariance matrix out of a randomized gaussian distribution
# Written by Blake, 9/14/2022

matrix = np.random.rand(100,100)
newmatrix = matrix
##
# newmatrix[25:30,25:30] = 10
# newmatrix[25:30,75:80] = 10
# newmatrix[60:80,40:60] = 10
plt.matshow(newmatrix)
plt.xlabel("X-Axis ")
plt.ylabel("Y-Axis")
#plt.show()

#Fun part: trying to propagate an orbit
#Using the TLE of the ISS, propagate its orbit for a whole day/month/etc.

# 1 25544U 98067A   22258.32873485  .00008862  00000+0  16183-3 0  9994
# 2 25544  51.6423 250.0020 0002297 234.9336 199.8491 15.50219806359164

i = 51.6423*3.1415/180     #rads
OMEGA = 250.0020*3.1415/180 #rads
e = 0.002297 
w = 234.9336*3.1415/180     #rads
M = 199.8491*3.1415/180     #rads
n = 15.50219806             #mean motion
t_span = np.linspace(0,864000,864000)
mu = 3.986*10**14 #units in m^3/s^2

#solving kepler's problem
m = 0
E = 0.5
diff = 1
while diff > 10**-7:
    mi = E - e*sin(E)
    diff = m - mi
    dm = 1 - e*cos(E)
    Ei = E + (diff)/(dm)
    diff = abs(m - mi)
    E = Ei
    
global f
f = 2*atan((sqrt((1+e)/(1-e))*tan(E/2)))
#convert orbital elements to a vector
def eltovec():
    mu = 3.986*10**14 #units in m^3/s^2
    i = 51.6423*3.1415/180     #rads
    OMEGA = 250.0020*3.1415/180 #rads
    e = 0.002297 
    w = 234.9336*3.1415/180     #rads
    M = 199.8491*3.1415/180     #rads
    n = 15.50219806             #mean motion
    a = (mu**(1/3))/((2*3.14159*n)/86400)**(2/3)
    print(a)
    global r,v,R
    mu = 3.986*10**14
    rp = a*(1-e) #perigee radius
    vp = sqrt(mu*((2/rp) - (1/a))) #perigee velocity
    h = rp*vp
    r = (((h**2)/(mu))/(1 + e*cos(f)))
    latus = a*(1-e**2)
    P = r*cos(f)
    Q = r*sin(f)
    perifocal = ([P],[Q],[0])
    R11 = cos(OMEGA)*cos(w) - sin(OMEGA)*sin(w)*cos(i)
    R12 = -cos(OMEGA)*sin(OMEGA) - sin(OMEGA)*cos(w)*cos(i)
    R13 = sin(OMEGA)*sin(i)
    R21 = sin(OMEGA)*cos(w) + cos(OMEGA)*sin(w)*cos(i)
    R22 = -sin(OMEGA)*sin(w) + cos(OMEGA)*cos(w)*cos(i)
    R23 = -cos(OMEGA)*sin(i)
    R31 = sin(w)*sin(i)
    R32 = cos(w)*sin(i)
    R33 = cos(i)
    R = np.array([[R11,R12,R13],[R21,R22,R23],[R31,R32,R33]])
    r = R@perifocal
    #now to find the velocity
    P = sqrt((mu)/(latus)*(-sin(f)))
    Q = sqrt((mu)/(latus)*(e + cos(f)))
    v = sqrt(P**2 + Q**2)
    perifocalv = ([P],[Q],[0])
    
    v = R@perifocalv
    return r,v
eltovec()

def orbdyn(x,t):
    mu = 3.986*10**14 #units in m^3/s^2
    Re = 6378135.00     #Radius of Earth in meters
    J2 = 0.00108263  #J2 Parameter for Earth Obliqueness
    r2 = sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    Coeff = (-3/2*J2*(mu/r2**2)*(Re/r2)**2)
    dx = np.zeros([6,1], dtype = float)
    dx[0] = x[3]
    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = (-mu/r2**3)*x[0] 
    dx[4] = (-mu/r2**3)*x[1] 
    dx[5] = (-mu/r2**3)*x[2] 
    dx = dx.reshape(6,)
    return dx
y0 = np.concatenate((r,v))
y0 = y0.reshape(6,)
solution = scipy.integrate.odeint(orbdyn,y0,t_span)
#plotting Earth
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig.add_subplot(111, projection='3d')

# Radii corresponding to the coefficients:
rx = 6378135
ry = 6378135
rz = 6371009

# Set of all spherical angles:
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
x = rx * np.outer(np.cos(u), np.sin(v))
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones_like(u), np.cos(v))

# Plot:
ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')

# Adjustment of the axes, so that they all have the same span:
max_radius = max(rx, ry, rz)
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
plt.show()

#plotting the orbit
ax = plt.axes(projection='3d')
ax.plot3D(solution[:,1], solution[:,2], solution[:,3]*1000, 'red')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()





