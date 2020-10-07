import numpy as np
from scipy import linalg
from ksvd import KSVD
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def ksvd_random_signal1(): # random signal test
    #A = np.random.randn(1, 60)                         #Senial      1 X 60
    solar = np.genfromtxt('savegol.csv',delimiter=',')
    #Reshape to (100,1)
    solar =np.array(solar[0:1000]);
    print('- - - - - -  solar : ', solar.shape)
    #Normalizar
    solar = np.reshape(solar,(solar.shape[0],1))
    Y = normalize(solar, axis=0, norm='max')
    print('- - - - - - solar 2 : ', Y.shape)

    #Y_a = np.random.normal(0.,1., (100,1))    #Dummie signal
    plt.plot(Y,label='Datos objetivo', s)
    plt.ylabel("Adjusted Flux (Solar Flux Unit (SFU))")
    plt.xlabel("Time Step (samples)")
    ksvd = KSVD(rank=1000,num_of_NZ=200)
    A, X= ksvd.fit(Y)
    return A, X

A,X = ksvd_random_signal1()
#ksvd_image('data/source/Lenna.jpg',100)
