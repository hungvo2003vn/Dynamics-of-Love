import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tfdiffeq import odeint
from scipy.signal import savgol_filter

path="D:\PYTHON\Dynamics of Love\exact.xlsx"

data=pd.read_excel(path)
data=pd.DataFrame(data)

R_ori=np.array(data['R'])
J_ori=np.array(data['J'])
# Clean noise
R = np.array(savgol_filter(R_ori, 51, 3))
J = np.array(savgol_filter(J_ori, 51, 3))

#Function Return R', J'
def parametric_ode_system(t, u, args):
    a, b, c, d = args[0], args[1], args[2], args[3]
    x, y = u[0], u[1]
    dx_dt = a*x + b*y 
    dy_dt = c*x + d*y 
    return tf.stack([dx_dt, dy_dt])

#Plot factors
t_begin=0.
t_end=0.999
t_nsamples=1000
t_space = np.linspace(t_begin, t_end, t_nsamples)

#Real Solution
dataset_outs = [tf.expand_dims(R, axis=1), tf.expand_dims(J, axis=1)]

#Trained
coef_200th=[3.0679794202251327, 1.2378169355916806, 3.1586576672319198, 0.3396771726693771]
coef_73th=[2.1260304454762737, 3.836140079929365, 5.757928441027235, -2.676600144513414]
coef_513th=[2.113135986365123, 3.8533390208093627, 5.783574932348526, -2.71142115818094]
# 60.15933188704432
#Learned parameters: [2.113135986365123, 3.8533390208093627, 5.783574932348526, -2.71142115818094]

#Initial value 
t_space_tensor = tf.constant(t_space)
R_init = tf.constant([-2.], dtype=t_space_tensor.dtype)
J_init = tf.constant([3.], dtype=t_space_tensor.dtype)
u_init = tf.convert_to_tensor([R_init, J_init], dtype=t_space_tensor.dtype)
coef_init = [1.0, 1.0, 1.0, 1.0]
args = [tf.Variable(initial_value=coef_init[i], name='p' + str(i+1), trainable=True, dtype=t_space_tensor.dtype) for i in range(0, 4)]

#More Coefficients:
learning_rate = 0.05
epochs = 200 #  Should change up to 5000
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#Solve the ODE sys for each coefficients:
def net():
  f=lambda ts, u0: parametric_ode_system(ts, u0, args)
  return odeint(f, u_init, t_space_tensor)

#Local Error Reducing:
def loss_func(num_sol):
  return tf.reduce_sum(tf.square(dataset_outs[0] - num_sol[:, 0])) + \
         tf.reduce_sum(tf.square(dataset_outs[1] - num_sol[:, 1]))


#Neural Network
for epoch in range(epochs):
  with tf.GradientTape() as tape:
    num_sol = net()
    loss_value = loss_func(num_sol)

  print("Epoch:", epoch, " loss:", loss_value.numpy())

  grads = tape.gradient(loss_value, args)
  optimizer.apply_gradients(zip(grads, args))


print("Learned parameters:", [args[i].numpy() for i in range(0, 4)])

num_sol = net()
R_num_sol = num_sol[:, 0].numpy()
J_num_sol = num_sol[:, 1].numpy()


plt.figure()
plt.plot(t_space, R_ori,'.', linewidth=1, label='Exact R')
plt.plot(t_space, J_ori,'.', linewidth=1, label='Exact J')
plt.plot(t_space, R_num_sol, linewidth=2, label='Approximate R')
plt.plot(t_space, J_num_sol, linewidth=2, label='Approximate J')
plt.title('Neural ODEs to fit params\n'+
          "R(0) = -2, J(0) = 3\n" +
          "R' = "+str(args[0].numpy())+"*R + "+str(args[1].numpy())+"*J\n"+
          "J' = "+str(args[2].numpy())+"*R + "+str(args[3].numpy())+"*J"
         )
plt.xlabel('Time, '+'h = 0.001')
plt.ylabel('Value of R(t), J(t) at t')
plt.grid(True)
plt.legend()
plt.show()



plt.figure()
#Plot exact R and Approximate R
plt.subplot(1,2,1)
plt.plot(t_space, R_ori,'.', linewidth=1, label='Exact R')
plt.plot(t_space, R_num_sol, linewidth=2,c='r', label='Approximate R')
plt.xlabel('Time, '+'h = 0.001')
plt.ylabel('Value of R(t) at t')
plt.title('exact R and approximate R')
plt.grid(True)
plt.legend()

#Plot exact J and Approximate J
plt.subplot(1,2,2)
plt.plot(t_space, J_ori,'.', linewidth=1, label='Exact J')
plt.plot(t_space, J_num_sol, linewidth=2,c='r', label='Approximate J')
plt.xlabel('Time, '+'h = 0.001')
plt.ylabel('Value of J(t) at t')
plt.title('exact J and approximate J')
plt.grid(True)
plt.legend()

plt.suptitle('Neural ODEs to fit params\n'+
          "R(0) = -2, J(0) = 3\n" +
          "R' = "+str(args[0].numpy())+"*R + "+str(args[1].numpy())+"*J\n"+
          "J' = "+str(args[2].numpy())+"*R + "+str(args[3].numpy())+"*J"
         , fontsize=7)
plt.show()


