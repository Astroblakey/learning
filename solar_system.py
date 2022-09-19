#Import all the packages needed
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.integrate import odeint
from numpy import sqrt
import math
from math import *
import sys

TLEs = []
with open("planets.txt", "r") as fd:
    lines = fd.readlines()

    # Loop through all lines, ignoring header.
    # Add last element to list 
    for l in lines[2::3]:
        TLEs.append(l.split())
values = np.array(TLEs)
lines = len(values)

fig = plt.figure(1)
ax = plt.axes(projection='3d')

for x in range(lines):

    global i,OMEGA,e,w,M,n,t_span,mu,f
    i = np.radians(float(values[x][2]))
    OMEGA = np.radians(float(values[x][3]))
    e = float(values[x][4])*0.0000001
    w = np.radians(float(values[x][5]))
    M = np.radians(float(values[x][6]))
    n = float(values[x][7])
    t_span = np.linspace(0,24624000,246240)    #time span in seconds, (start, stop, #elements)
    mu = 1.3271*10**20                    #units in m^3/s^2

    # Solving kepler's problem (using a Fourier expression)
    E = M + (e - (1/8)*e**3)*sin(M) + ((1/2)*e**2)*sin(2*M) + ((3/8)*e**3)*sin(3*M)
    f = 2*atan((sqrt((1+e)/(1-e))*tan(E/2)))
    #convert orbital elements to vectors
    def eltovec():
        #Calculate values needed to transform to perifocal frame
        a = (mu**(1/3))/((2*3.14159*n)/86400)**(2/3)
        global rECI,vECI
        rp = a*(1-e)                    #perigee radius
        vp = sqrt(mu*((2/rp) - (1/a)))  #perigee velocity
        h = rp*vp
        r = (((h**2)/(mu))/(1 + e*cos(f)))
        latus = a*(1-e**2)
        P = r*cos(f)
        Q = r*sin(f)
        perifocal = ([P],[Q],[0])
        R11 = cos(OMEGA)*cos(w) - sin(OMEGA)*sin(w)*cos(i)
        R12 = -cos(OMEGA)*sin(w) - sin(OMEGA)*cos(w)*cos(i)
        R13 = sin(OMEGA)*sin(i)
        R21 = sin(OMEGA)*cos(w) + cos(OMEGA)*sin(w)*cos(i)
        R22 = -sin(OMEGA)*sin(w) + cos(OMEGA)*cos(w)*cos(i)
        R23 = -cos(OMEGA)*sin(i)
        R31 = sin(w)*sin(i)
        R32 = cos(w)*sin(i)
        R33 = cos(i)
        R = np.array([[R11,R12,R13],[R21,R22,R23],[R31,R32,R33]])
        r = R@perifocal

        #Convert perifocal r to ECI r
        R3O = np.array([[cos(OMEGA),sin(OMEGA),0],[-sin(OMEGA),cos(OMEGA),0],[0,0,1]])
        R1 = np.array([[1,0,0],[0,cos(i),sin(i)],[0,-sin(i),cos(i)]])
        R3o = np.array([[cos(w),sin(w),0],[-sin(w),cos(w),0],[0,0,1]])
        rECI = R3O@R1@R3o@r

        #now to find the velocity
        P = sqrt((mu)/(latus))*(-sin(f))
        Q = sqrt((mu)/(latus))*(e + cos(f))
        v = sqrt(P**2 + Q**2)
        perifocalv = ([P],[Q],[0])
        v = R@perifocalv
        vECI = R3O@R1@R3o@v
        vECI = np.vstack(vECI)
        return rECI,vECI
    eltovec()

    #Orbit Propagator
    def orbdyn(x,t):  #units in m^3/s^2
        Re = 695700000    #Radius of Earth in meters
        J2 = 0.00108263     #J2 Parameter for Earth Obliqueness
        dx = np.zeros([6,1], dtype = float)
        i = x[0]
        j = x[1]
        k = x[2]
        r2 = (i**2 + j**2 + k**2)
        Coeff = (-3/2*J2*(mu/r2**2)*(Re/r2)**2)
        xdot = x[3]
        ydot = x[4]
        zdot = x[5]
        xddot = -mu * i/ (r2) ** (3/2) + Coeff * (1-5 * ((x[2]/r2) ** 2)) * (x[0]/r2)
        yddot = -mu * j/ (r2) ** (3/2) + Coeff * (1-5 * ((x[2]/r2) ** 2)) * (x[1]/r2)
        zddot = -mu * k/ (r2) ** (3/2)  + Coeff * (3-5 * ((x[2]/r2) ** 2)) * (x[2]/r2)
        dstated = [xdot, ydot, zdot, xddot, yddot, zddot]
        return dstated

    #state initial conditions
    X_0 = rECI[0]  # [m]
    Y_0 = rECI[1]   # [m]
    Z_0 = rECI[2]   # [m]
    VX_0 = vECI[0]  # [m/s]
    VY_0 = vECI[1]   # [m/s]
    VZ_0 = vECI[2]   # [m/s]

    state_0 = [X_0, Y_0, Z_0, VX_0, VY_0, VZ_0]
    state_0 = np.vstack(state_0)
    state_0 = state_0.reshape(6,) #for some reason it needs to be 1-D, so use reshape to allow that

    #solving system
    solution = odeint(orbdyn,state_0,t_span) #solve using odeint
    x_pos = solution[:,0]
    y_pos = solution[:,1]
    z_pos = solution[:,2]

    # Setting up Spherical Earth to Plot
    N = 100
    phi = np.linspace(0, 2 * np.pi, N)
    theta = np.linspace(0, np.pi, N)
    theta, phi = np.meshgrid(theta, phi)

    r_Earth = 695700000  # Average radius of sun (m)
    X_Earth = r_Earth * np.cos(phi) * np.sin(theta)
    Y_Earth = r_Earth * np.sin(phi) * np.sin(theta)
    Z_Earth = r_Earth * np.cos(theta)

    # Plotting Earth and Orbit
    ax.plot_surface(X_Earth, Y_Earth, Z_Earth, color='yellow', alpha=0.99)
    ax.plot3D(x_pos, y_pos, z_pos, 'red')
    ax.view_init(30, 145)  # Changing viewing angle (adjust as needed)
    plt.title('Two-Body Orbit')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    # Make axes limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(),      
                    ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim * 3/4)

plt.show()

