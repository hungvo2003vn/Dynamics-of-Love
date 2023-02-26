import numpy as np
import pandas as pd
import platform
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import scipy.integrate as spi
import math

#############################################################
#  Input:
#
#    function f: evaluates the right hand side of the ODE.  
#
#    real tspan[2]: the starting and ending times.
#
#    real y0[m]: the initial conditions. 
#
#    integer n: the number of steps.
#
#  Output:
#
#    real t[n+1], y[n+1,m]: the solution estimates.
def backward_euler ( f, tspan, y0, n ):

  # Checking initial value of R[0], J[0]:
  if ( np.ndim (y0) == 0 ):
    m = 1
  else:
    m = len (y0)

  # list of t and [R,J]:
  t = np.zeros (n)
  y = np.zeros ( [ n , m ] )

  #step's value calculation:
  dt = ( tspan[1] - tspan[0] ) / float ( n )

  # Initial t[0] and R[0], J[0]:
  t[0] = tspan[0]
  y[0,:] = y0

  # Estimating R[ti], J[ti] using implicit Euler's method:
  for i in range (0, n-1):

    to = t[i]
    yo = y[i,:]
    t1 = t[i] + dt

    # The starting estimate for the roots of func(x) = 0
    # Explicit estimation putting in the fsolve to make it find y1 find faster
    y1 = yo + dt * f (to, yo) 

    y1 = fsolve( backward_euler_residual, y1, args = ( f, to, yo, t1 ) )

    # Store t[i] and y[i] in the list
    t[i+1]   = t1
    y[i+1,:] = y1[:]

  return t, y
#############################################################
def backward_euler_residual ( y1, f, to, yo, t1 ):

  value = y1 - yo - ( t1 - to ) * f ( t1, y1 )

  return value
############## 5 Examples of using Implicit Euler's Method ##############
####################### Model's Initialization #######################
#  Input:
#
#    real T, the current time.
#
#    real RF[2], the current solution variables, rabbits and foxes.
#
#  Output:
#
#    real DRFDT[2], the right hand side of the 2 ODE's.
####################### 1st Model #######################
def First(t, RJ):
  # Value of R, J at t:
  R = RJ[0]
  J = RJ[1]

  # Non linear system ODE calculating f(t,R[t], J[t]):
  dRdt = R * (1 - R) - R * J
  dJdt = 2 * J * (1 - (J**2)/2) - 3 * (R**2) * J

  # Matrix of R'[t], J'[t]:
  dRJdt = np.array ( [ dRdt, dJdt ] )

  return dRJdt
####################### 2nd Model #######################
def Second(t, RJ):
  # Value of R, J at t:
  R = RJ[0]
  J = RJ[1]

  # Non linear system ODE calculating f(t,R[t], J[t]):
  dRdt = R * (1 - R - J)
  dJdt = 2 * J * (1 - J/2 - 3/2 * R)

  # Matrix of R'[t], J'[t]:
  dRJdt = np.array ( [ dRdt, dJdt ] )

  return dRJdt
####################### 3rd Model #######################
def Third(t, RJ):
  # Value of R, J at t:
  R = RJ[0]
  J = RJ[1]

  # Non linear system ODE calculating f(t,R[t], J[t]):
  dRdt = -R**2 + 1
  dJdt = R * (J - 1)

  # Matrix of R'[t], J'[t]:
  dRJdt = np.array ( [ dRdt, dJdt ] )

  return dRJdt
####################### 4th Model #######################
def Fourth(t, RJ):
  # Value of R, J at t:
  R = RJ[0]
  J = RJ[1]

  # Non linear system ODE calculating f(t,R[t], J[t]):
  dRdt = R**2 - J**2 - 1
  dJdt = R - J + 2

  # Matrix of R'[t], J'[t]:
  dRJdt = np.array ( [ dRdt, dJdt ] )

  return dRJdt
####################### 5th Model #######################
def Fifth(t, RJ):
  # Value of R, J at t:
  R = RJ[0]
  J = RJ[1]

  # Non linear system ODE calculating f(t,R[t], J[t]):
  dRdt = 2 * R - R * J
  dJdt = -9 * J + 3 * R * J

  # Matrix of R'[t], J'[t]:
  dRJdt = np.array ( [ dRdt, dJdt ] )

  return dRJdt
####################### Predator-Prey's Model #######################
def predator_prey_deriv ( t, RJ ):

  # Value of R, J at t:
  R = RJ[0]
  J = RJ[1]

  # Non linear system ODE calculating f(t,R[t], J[t]):
  dRdt =    2.0 * R - 0.001 * R * J
  dJdt = - 10.0 * J + 0.002 * R * J

  # Matrix of R'[t], J'[t]:
  dRJdt = np.array ( [ dRdt, dJdt ] )

  return dRJdt
####################### Visualization Model #######################
#  Input:
#    f: function of the model
#    real tspan[2]: the time span
#
#    real y0[2]: the initial condition.
#     
#    integer n: the number of steps to take.
#
#    number: to know where the figure's position of this model is
def Visual (f, tspan, y0, n, number):

  step = (tspan[1]-tspan[0])/n # Time's step
  t, y = backward_euler(f, tspan, y0, n)
  exact_solution = spi.solve_ivp(f, tspan, y0, method='RK45', t_eval = t)
  y_exact = pd.DataFrame(exact_solution.y).T
  t_exact = pd.DataFrame(exact_solution.t)

  plt.clf() # Clear old Model's figure
  # Plot R[t]:
  plt.subplot(1,2,1)
  plt.plot(t, y[:,0], 'r-', linewidth = 1, label = 'Approximate R')
  plt.plot(t_exact, y_exact[:][0], 'y-', linewidth = 1, label = 'Exact R')
  plt.grid(True)
  plt.xlabel('<--- t --->')
  plt.ylabel('<--- R(t) --->')
  plt.legend()
  plt.title('Exact R and Approxiamte R')

  # Plot J[t];
  plt.subplot(1,2,2)
  plt.plot(t, y[:,1], 'b-', linewidth = 1, label = 'Approximate J')
  plt.plot(t_exact, y_exact[:][1], 'y-', linewidth = 1, label = 'Exact J')
  plt.grid(True)
  plt.xlabel('<--- t --->')
  plt.ylabel('<--- J(t) --->')
  plt.legend()
  plt.title('Exact J and Approxiamte J')

  # General title:
  plt.suptitle('Model ' + str(number)+ ' solved by ODE backward Euler\'s method')
  plt.show()

  '''
  filename = "Model" + str(number) +" solved by Implicit Euler.png"
  plt.savefig(filename)
  print('  Graphics saved as "%s"' % ( filename ))
  plt.show(block = False)
  plt.close()
  '''

  return
####################### All Model #######################
def backward_euler_test ():

  print('')
  print('backward_euler_test():')

  ####Testing the all Model:
  ## Initial condition
  tspan = np.array ( [ 0.0, 5.0 ] )
  y0 = np.array ( [ 2, 3 ] )
  n = 200
  
  ## Visualization:
  # 1st Model:
  Visual( First, tspan, y0, n, 1)
  # 2nd Model:
  Visual( Second, tspan, y0, n, 2)
  # 3rd Model:
  Visual( Third, tspan, y0, n, 3 )
  # 4th Model:
  Visual( Fourth, tspan, y0, n, 4 )
  # 5th Model:
  Visual( Fifth, tspan, y0, n, 5 )

  plt.show()
  print('')
  print('  Normal end of execution.')
  return
########################### MAIN ###########################

if ( __name__ == '__main__' ):
  backward_euler_test()
  
