import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
Testing Function for forwarding and backwarding Euler's method:
y' = -10y with t0 = 0, y(0) = 1 
Exact solution is 
    y(t) = e^(-10t) * e^c, e^c = 1
'''
#Initial coefficients
h_exact=0.01
h_app=0.21
t0=0
y0=1
p=-10
N=10
alpha=10
#List of all step
x_exact=[t0+i*h_exact for i in range(alpha*N)]
x_app=[t0+i*h_app for i in range(N)]

#derative function y = e^(-10t)
def F(t):
    return y0*(np.exp(p*t))

#derative function approximate y'=lambda*y
def f(t,y_t): 
    return p*y_t

#Forward Euler's function:
def forwardEuler(t_k, y_k, h):

    y_k1 = y_k + f(t_k,y_k)*h
    return y_k1
#Backward Euler's function:
def BackwardEuler(t_k, y_k, h):

    y_k1 = y_k*(1/(1-p*h)) 
    return y_k1


#Exact solution:
Y_exact=[F(x_exact[i]) for i in range(alpha*N)]

# Approximating solution using Explicit Euler's method:
Y_app_explicit=[y0]
Y_app_implicit=[y0]

for i in range (N-1):
    Y_app_explicit+=[forwardEuler(x_app[i], Y_app_explicit[i], h_app)]
    Y_app_implicit+=[BackwardEuler(x_app[i], Y_app_implicit[i], h_app)]


plt.plot(x_exact, Y_exact, color='g', label='Exact solution')
plt.plot(x_app,Y_app_explicit, color='r', label='Explicit Euler')
plt.plot(x_app,Y_app_implicit, color='b', label='Implicit Euler')

plt.xlabel("Time, h = "+str(h_app))
plt.ylabel('Value Y(t) at time t')
plt.title("Solution for y' = "+str(p)+"y, y(0) = "+str(y0))

plt.grid(True)
plt.legend()
#plt.show()

filename = "Comparision between implicit and explicit Euler's method.png"
plt.savefig(filename)
print('  Graphics saved as "%s"' % ( filename ))
plt.show(block = False)
plt.close()