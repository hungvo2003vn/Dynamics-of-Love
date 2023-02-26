import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.signal import savgol_filter

path="D:\PYTHON\Dynamics of Love\exact.xlsx"

data=pd.read_excel(path)
data=pd.DataFrame(data)

R=np.array(data['R'])
J=np.array(data['J'])

#Plot factors
t_begin=0.
t_end=0.999
t_nsamples=1000
t_space = np.linspace(t_begin, t_end, t_nsamples)

window, degree = 7, 3
R_cleaned = savgol_filter(R, window, degree)
J_cleaned = savgol_filter(J, window, degree)

#Plot R and R cleaned
plt.subplot(2,2,1)
plt.plot(t_space, R_cleaned, linewidth=1, linestyle="-", c="b",label='Cleaned R')
plt.xlabel('times')
plt.ylabel('R(t)')
plt.title('Cleaned R')
plt.grid('t')

plt.subplot(2,2,2)
plt.plot(t_space, R,'-', linewidth=1,c='r', label='Noise R')

plt.xlabel('times')
plt.ylabel('R(t)')
plt.title('Noise R')
plt.grid('t')


#Plot J and J cleaned
plt.subplot(2,2,3)
plt.plot(t_space, J_cleaned, linewidth=1, linestyle="-", c="b",label='Cleaned J')
plt.xlabel('times')
plt.ylabel('J(t)')
plt.title('Cleaned J')
plt.grid('t')

plt.subplot(2,2,4)
plt.plot(t_space, J,'-', linewidth=1,c='r', label='Noise J')

plt.xlabel('times')
plt.ylabel('J(t)')
plt.title('Noise J')
plt.grid('t')

plt.suptitle("Savitzky–Golay method with window = "+str(window)+", degree = "+str(degree))
plt.show()