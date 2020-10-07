import numpy as np
from ksvd import KSVD
import matplotlib.pyplot as plt
import csv


y = []
x = []
with open('savegol.csv') as csvfile:
    plots= csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in plots:
        y.append(float(row[0]))
                

plt.plot(y)
plt.xlabel('Time (Julian Date)')
plt.ylabel('adjusted_flux (solar flux unit (SFU))')
plt.show()

y=np.array(y)
n.resize(y, ())

print(y)



def ksvd_random_signal1(): # random signal test
    A = np.random.randn(1, 1000)                                                   #Senial      1 X 60
    Y_a = np.random.normal(0.,1., (1000,1))                                         #Dictionary  60 X 4000
    plt.plot(Y_a,label='Datos objetivo')
    
    ksvd = KSVD(rank=1000,num_of_NZ=100)
    A, X= ksvd.fit(Y_a)

    
    return A, X


A,X = ksvd_random_signal1()

#ksvd_image('data/source/Lenna.jpg',100)

