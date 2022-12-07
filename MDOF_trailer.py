#!/usr/bin/env python
# coding: utf-8

# # MEC4047F Assignment 2

# In[2]:


#imports
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import scipy
from scipy import signal


# In[3]:


#CONSTANTS

m1 = 18.926
m2 = 18.926
m3 = 604.812
I = 600*(0.7**2)
c_s = 4e3
k_s = 45e3
k_t = 200e3
s = 760e-3
d_o = 20e-3
d_i = 14e-3
h = 250e-3
v = 0.3
E = 200e9
G = E/(2*(1+v))
J = (math.pi/32)*(d_o**4-d_i**4)
k_tor = (G*J)/(s*h**2)


# In[68]:


#MATRICES
DOF = 4
M = np.array([[m1, 0 , 0, 0], [0, m2, 0, 0],[0, 0, m3, 0], [0, 0, 0, I]])
C = np.array([[c_s, 0, -c_s, s*c_s], [0, c_s, -c_s, -s*c_s], [-c_s, -c_s, 2*c_s, 0], [s*c_s, -s*c_s, 0, 2*s**2*c_s]])
K = np.array([[(k_t+k_s+k_tor), -k_tor, -k_s, s*(k_s+2*k_tor)], [-k_tor, (k_t+k_s+k_tor), -k_s, -s*(k_s+2*k_tor)], [-k_s, -k_s, 2*k_s, 0],[s*(k_s+2*k_tor), -s*(k_s+2*k_tor), 0, s**2*(2*k_s+4*k_tor)]])
print(M)
print(C)
print(K)


# In[5]:


#RAYLEIGH DAMPING
a = np.array([[M[0,0], K[0,0]],[M[0,1], K[0,1]]])
b = np.array([C[0,0], C[0,1]])
x = np.linalg.solve(a,b)

C_new = x[0]*M + x[1]*K


# In[6]:


#EIGENSTUFF
w, v = scipy.linalg.eig(K,M, right = True)
for i in v:
    for j in i:
        j = round(j,4)
print("Eigenvalues:",w)
for i in range(4):
    print("Mode",i,v[i])

                                     


# # CENTRAL DIFFERENCE APPROXIMATION - IN PHASE

# In[69]:


#CENTRAL DIFFERENCE APPROXIMATION - IN PHASE BUMPS

#time
time = 2
h = 0.002
n = int(time/h)
t = np.linspace(0, time, n+1) #-1 to n -> i=1 is 0th element

#forcing functions
F = np.zeros((DOF,n+1))
v = 10.41301627
#v=80/3.6
L = 6
omega = 2*math.pi*(v/L)
y_l = np.zeros(len(t))
for i in range(len(t)):
    y_l[i] = 0.05*math.sin(omega*t[i])
    
y_r = np.zeros(len(t))
for i in range(len(t)):
    y_r[i] = 0.05*math.sin(omega*t[i]) #in phase with y_l

f_l = k_t*y_l
f_r = k_t*y_r
F[0] = f_l
F[1] = f_r


#initialize solution
z = np.zeros((DOF,n+1))
v = np.zeros((DOF,n+1))
a = np.zeros((DOF,n+1))

for i in range(DOF-1): #initial velocity and position
    z[i+1][0] = 0
    v[i+1][0] = 0    
a[:,1] = np.matmul(np.linalg.inv(M),(F[:,1]-np.matmul(C,v[:,1])-np.matmul(K,z[:,1]))) #a_0
z[:,0] = z[:,1] - h*v[:,1] + (h**2)/2 * a[:,1] #z_-1

#precompute constants
A_1 = M/(h**2) #nxn
A_2 = C/(2*h) #nxn
A_3 = A_1+A_2
A_4 = A_1-A_2
A_3inv = np.linalg.inv(A_3)

#CDA scheme 
for i in range(len(z[0])-1):
    z[:,i+1] = np.matmul(A_3inv,F[:,i]) - np.matmul(np.matmul(A_3inv,K-2*A_1),z[:,i]) - np.matmul(np.matmul(A_3inv,A_4),z[:,i-1])


#plot
def annot_max(x,y,x_loc,y_loc, ax=None): # Function for annotating the max values
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.1)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="bottom")
    ax.annotate(text, xy=(xmax, ymax), xytext=(x_loc,y_loc), **kw)
    
plt.figure(figsize=(10,8))
plt.plot(t,y_l, label = "Base Excitation")
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title("Base Excitation")
plt.show

plt.figure(figsize=(10,8))
plt.plot(t, z[0], label = "Left Wheel")
plt.plot(t, z[1], label = "Right Wheel")
plt.plot(t, z[2], label = "Trailer")
plt.plot(t, z[3], label = "Trailer Roll")
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title("Analytical Solution")
plt.legend(loc="upper right")
annot_max(t,z[2],0.5,0.97)
plt.show()


# # CENTRAL DIFFERENCE APPROXIMATION - OUT OF PHASE

# In[70]:


#CENTRAL DIFFERENCE APPROXIMATION - OUT OF PHASE BUMPS

#time
time = 2
h = 0.002
n = int(time/h)
t = np.linspace(0, time, n+1) #-1 to n -> i=1 is 0th element

#forcing functions
F = np.zeros((DOF,n+1))
v = 40/3.6
L = 6
phase = math.pi*0.25 #phase angle
omega = 2*math.pi*(v/L)
y_l = np.zeros(len(t))
for i in range(len(t)):
    y_l[i] = 0.05*math.sin(omega*t[i]+phase)
    
y_r = np.zeros(len(t))
for i in range(len(t)):
    y_r[i] = 0.05*math.sin(omega*t[i])

f_l = k_t*y_l
f_r = k_t*y_r
F[0] = f_l
F[1] = f_r


#initialize solution
z = np.zeros((DOF,n+1))
v = np.zeros((DOF,n+1))
a = np.zeros((DOF,n+1))

for i in range(DOF-1): #initial velocity and position
    z[i+1][0] = 0
    v[i+1][0] = 0    
a[:,1] = np.matmul(np.linalg.inv(M),(F[:,1]-np.matmul(C,v[:,1])-np.matmul(K,z[:,1]))) #a_0
z[:,0] = z[:,1] - h*v[:,1] + (h**2)/2 * a[:,1] #z_-1

#precompute constants
A_1 = M/(h**2) #nxn
A_2 = C/(2*h) #nxn
A_3 = A_1+A_2
A_4 = A_1-A_2
A_3inv = np.linalg.inv(A_3)

#CDA scheme 
for i in range(len(z[0])-1):
    z[:,i+1] = np.matmul(A_3inv,F[:,i]) - np.matmul(np.matmul(A_3inv,K-2*A_1),z[:,i]) - np.matmul(np.matmul(A_3inv,A_4),z[:,i-1])

#plot
plt.figure(figsize=(10,8))
plt.plot(t,y_l, label = "Left Base Excitation")
plt.plot(t,y_r, label = "Right Base Excitation")
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title("Base Excitation - Phase angle of")
plt.legend(loc="upper right")
plt.show()

plt.figure(figsize=(10,8))
plt.plot(t, z[0], label = "Left Wheel")
plt.plot(t, z[1], label = "Right Wheel")
plt.plot(t, z[2], label = "Trailer")
#plt.plot(t, z[3], label = "Trailer Roll")
plt.title("Central Difference Approximation - Phase angle of 0.25*pi")
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.legend(loc="upper right")
plt.show

plt.figure(figsize=(10,8))
plt.plot(t, z[3], label = "Trailer Roll")
plt.title("Central Difference Approximation - Phase angle of 0.25*pi")
plt.legend(loc="upper right")
plt.xlabel('time (s)')
plt.ylabel('displacement (radians)')
plt.legend(loc="upper right")
plt.show


# # DISPLACEMENT MAXIMIZATION

# In[ ]:


velocities = np.linspace(0,80,800)
max_displacements = np.zeros(800)
k=0
for v in velocities:   
    
    #forcing functions
    F = np.zeros((4,n+1))
    L = 6
    omega = 2*math.pi*(v/L)
    y_l = np.zeros(len(t))
    for i in range(len(t)):
        y_l[i] = 0.05*math.sin(omega*t[i])
    
    y_r = np.zeros(len(t))
    for i in range(len(t)):
        y_r[i] = 0.05*math.sin(omega*t[i]) #in phase with y_l

    f_l = k_t*y_l
    f_r = k_t*y_r
    F[0] = f_l
    F[1] = f_r


    #initialize solution
    z = np.zeros((4,n+1))
    v = np.zeros((4,n+1))
    a = np.zeros((4,n+1))

    for i in range(3): #initial velocity and position
        z[i+1][0] = 0
        v[i+1][0] = 0    
    a[:,1] = np.matmul(np.linalg.inv(M),(F[:,1]-np.matmul(C,v[:,1])-np.matmul(K,z[:,1]))) #a_0
    z[:,0] = z[:,1] - h*v[:,1] + (h**2)/2 * a[:,1] #z_-1

    #precompute constants
    A_1 = M/(h**2) #nxn
    A_2 = C/(2*h) #nxn
    A_3 = A_1+A_2
    A_4 = A_1-A_2
    A_3inv = np.linalg.inv(A_3)

    #CDA scheme 
    for i in range(len(z[0])-1):
        z[:,i+1] = np.matmul(A_3inv,F[:,i]) - np.matmul(np.matmul(A_3inv,K-2*A_1),z[:,i]) - np.matmul(np.matmul(A_3inv,A_4),z[:,i-1])
    max_displacement = max(z[2])
    max_displacements[k] = max_displacement
    k+=1
max_displacement = max(max_displacements)
print("Max Displacement = ",max_displacement)
index = np.where(max_displacements == max_displacement)
print("Velocity = ",velocities[index])
    


# # CODE VERIFICATION

# # FORCING FREQUENCY = PI

# In[73]:


#Matrices
DOF = 2
M = np.array([[18,0],[0,9]])
C = np.array([[0,0],[0,0]])
K = np.array([[192,-64],[-64,64]])

#time
time = 10
h = 0.01
n = int(time/h)
t = np.linspace(0, time, n+1) #-1 to n -> i=1 is 0th element

#Forcing function
F = np.zeros((DOF,n+1))
for i in range(n):
    F[0][i] = 10*math.cos(math.pi*t[i])
    

 #initialize solution
z = np.zeros((DOF,n+1))
v = np.zeros((DOF,n+1))
a = np.zeros((DOF,n+1))

#initial velocity and position
z[0][1] = 0
z[1][1] = 0
v[0][1] = 0    
v[1][1] = 0
a[:,1] = np.matmul(np.linalg.inv(M),(F[:,1]-np.matmul(C,v[:,1])-np.matmul(K,z[:,1]))) #a_0
z[:,0] = z[:,1] - h*v[:,1] + (h**2)/2 * a[:,1] #z_-1

#precompute constants
A_1 = M/(h**2) #nxn
A_2 = C/(2*h) #nxn
A_3 = A_1+A_2
A_4 = A_1-A_2
A_3inv = np.linalg.inv(A_3)

#CDA Scheme
for i in range(1,n):
    z[:,i+1] = np.matmul(A_3inv,(F[:,i]-np.matmul((K-2*A_1),z[:,i])-np.matmul(A_4,z[:,i-1])))
    
#analytical solution
analytical_1 = np.zeros((n+1))
analytical_2= np.zeros((n+1))
for i in range(n):
    analytical_1[i] = 0.0293*math.cos(math.sqrt(64/18)*t[i]) - 0.085*math.cos(math.sqrt(128/9)*t[i]) + 0.0557*math.cos(math.pi*t[i])
    analytical_2[i] = 0.0587*math.cos(math.sqrt(64/18)*t[i]) + 0.085*math.cos(math.sqrt(128/9)*t[i]) - 0.1437*math.cos(math.pi*t[i])
    
# Error at peak
z1Max = z[0].max()
solmax = analytical_1.max()
error = abs((z1Max-solmax)/solmax)
error = round(error*100,3)

#plot
plt.figure(figsize=(10,8))
plt.plot(t,z[0], label = "x1 numerical")
plt.plot(t,analytical_1, label = "x1 analytical")
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title("Analytical vs Numerical Solutions for x1")
#plt.title("Forcing Frequency = Twice The Largest Modal Frequency")
plt.legend(loc="upper right")
annot_max(t,z[0],0.5,0.97)
annot_max(t,analytical_1,0.5,0.94)
plt.text(0.1,0.1, "Time step ="+str(h)+"\nError ="+str(error)+"%")
plt.show()

plt.figure(figsize=(10,8))
plt.plot(t,z[1], label = "x2 numerical")
plt.plot(t,analytical_2, label = "x2 analytical")
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title("Analytical vs Numerical solutions for x2")
#plt.title("Forcing Frequency = Twice The Largest Modal Frequency")
plt.legend(loc="upper right")
plt.show()


# # FORCING FREQUENCY = 10

# In[81]:


#Matrices
DOF = 2
M = np.array([[18,0],[0,9]])
C = np.array([[0,0],[0,0]])
K = np.array([[192,-64],[-64,64]])

#time
time = 10
h = 0.001
n = int(time/h)
t = np.linspace(0, time, n+1) #-1 to n -> i=1 is 0th element

#Forcing function
F = np.zeros((DOF,n+1))
for i in range(n):
    F[0][i] = 10*math.cos(10*t[i])
    

 #initialize solution
z = np.zeros((DOF,n+1))
v = np.zeros((DOF,n+1))
a = np.zeros((DOF,n+1))

#initial velocity and position
z[0][1] = 0
z[1][1] = 0
v[0][1] = 0    
v[1][1] = 0
a[:,1] = np.matmul(np.linalg.inv(M),(F[:,1]-np.matmul(C,v[:,1])-np.matmul(K,z[:,1]))) #a_0
z[:,0] = z[:,1] - h*v[:,1] + (h**2)/2 * a[:,1] #z_-1

#precompute constants
A_1 = M/(h**2) #nxn
A_2 = C/(2*h) #nxn
A_3 = A_1+A_2
A_4 = A_1-A_2
A_3inv = np.linalg.inv(A_3)

#CDA Scheme
for i in range(1,n):
    z[:,i+1] = np.matmul(A_3inv,(F[:,i]-np.matmul((K-2*A_1),z[:,i])-np.matmul(A_4,z[:,i-1])))
    
#analytical solution
analytical_1 = np.zeros((n+1))
analytical_2= np.zeros((n+1))
for i in range(n):
    analytical_1[i] = (1/math.sqrt(27))*(0.00997*math.cos(math.sqrt(64/18)*t[i])+0.0224*math.cos(math.sqrt(128/9)*t[i])-0.0324*math.cos(10*t[i]))
    analytical_2[i] = (1/math.sqrt(27))*(0.0199*math.cos(math.sqrt(64/18)*t[i])-0.0224*math.cos(math.sqrt(128/9)*t[i])+0.0025*math.cos(10*t[i]))
    
# Error at peak
z1Max = z[0].max()
solmax = analytical_1.max()
error = abs((z1Max-solmax)/solmax)
error = round(error*100,3)

#plot
plt.figure(figsize=(10,8))
plt.plot(t,z[0], label = "x1 numerical")
plt.plot(t,analytical_1, label = "x1 analytical")
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title("Analytical vs Numerical Solutions for x1 - Forcing Frequency = 10 rad/s")
#plt.title("Forcing Frequency = Twice The Largest Modal Frequency")
plt.legend(loc="upper right")
annot_max(t,z[0],0.5,0.97)
annot_max(t,analytical_1,0.5,0.94)
plt.text(0.01,0.01, "Time step ="+str(h)+"\nError ="+str(error)+"%")
plt.show()

plt.figure(figsize=(10,8))
plt.plot(t,z[1], label = "x2 numerical")
plt.plot(t,analytical_2, label = "x2 analytical")
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title("Analytical vs Numerical solutions for x2 - Forcing Frequency = 10 rad/s")
#plt.title("Forcing Frequency = Twice The Largest Modal Frequency")
plt.legend(loc="upper right")
plt.show()


# In[ ]:




