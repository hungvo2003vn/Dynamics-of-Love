import ast
import numpy as np
import matplotlib.pyplot as plt
path="D:\\PYTHON\\Dynamics of Love\\Reduce_loss and Final result better version.txt" # path of the output.txt file

fp=open(path, 'r')
lines=fp.readlines()

def take_parameters(str):
    str = str[20:]
    return ast.literal_eval(str)

args = []
for i in range(len(lines)):
    if lines[i][:5] == 'Epoch':
        continue
    else:
        args+= [take_parameters(lines[i])]

n = len(args)
t = np.linspace(2760,5259,n)

params = ['a','b','c','d']
Colors = ['b-', 'r-', 'y-', 'g-']

#Best coeffs
coef = [2.113135986365123, 3.8533390208093627, 5.783574932348526, -2.71142115818094]
# plot 
plt.subplot(4,1,1)
for i in range(0,4):

    plt.subplot(4,1,i+1)
    tmp = [args[j][i] for j in range(n)]
    plt.plot(t, tmp,Colors[i], linewidth = 1, label = 'Coef '+params[i])
    if i == 3:
        plt.xlabel('<--- Epoch -->')
    plt.ylabel("Value of coeff")
    plt.legend()

plt.suptitle("Training session" + '\n' +
                'Best fit coefficients '+ '[' + str(coef[0])+', ' + str(coef[1])+', '+ str(coef[2])+', '+ str(coef[3])+']'
            )
plt.show()
