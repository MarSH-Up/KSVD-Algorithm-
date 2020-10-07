import numpy as np
from ksvd import KSVD
from numpy import random

Y=random.randint(100, size=(60,1000))
#Y=random.randn(60,1000)
ksvd = KSVD(rank=np.linalg.matrix_rank(Y),num_of_NZ=4)
A, X= ksvd.fit(Y)

"""
Y = AX
Y: shape = [n_features,n_samples]
A: Dictionary = [n_features, rank]
X: Sparse = [rank, n_samples]
"""
