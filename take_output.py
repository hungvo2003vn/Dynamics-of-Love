path="D:\\PYTHON\\Dynamics of Love\\Reduce_loss and Final result better version 3.txt" # path of the output.txt file

fp=open(path, 'r')
lines=fp.readlines()

def take_loss(str):
    idx=7
    while str[idx] != ':':
        idx+=1
    return float(str[idx+2:])

args=''
min_loss=10000
for i in range(len(lines)):
    if lines[i][:5] != 'Epoch':
        continue
    if min_loss > take_loss(lines[i]):
        min_loss=take_loss(lines[i])
        args=lines[i+1]
print(min_loss)
print(args)