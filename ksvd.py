import numpy as np
import scipy as sp
from sklearn.linear_model import OrthogonalMatchingPursuit
import matplotlib.pyplot as plt

class KSVD:
    def __init__(self,rank,num_of_NZ=None,func_svd=False,
                 max_iter = 5,max_tol = 1e-12):
        self.rank = rank
        self.max_iter = max_iter
        self.max_tol = max_tol
        self.num_of_NZ = num_of_NZ
        self.func_svd = func_svd

    def _initialize_parameters(self,Y):
        A = np.random.randn(Y.shape[0],self.rank)
        print('- ----- Diccionario creado con la forma: ', A.shape)
        X = np.zeros((self.rank,Y.shape[1]))
        print('- ---- Vector de dispersion creato de la forma : ', X.shape)
        return A, X

    def _estimate_X(self,Y,A):
        if self.num_of_NZ is None:
            n_nonzero_coefs = np.ceil(0.1 * A.shape[1])
        else:
            n_nonzero_coefs = self.num_of_NZ

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs = int(n_nonzero_coefs))
        for j in range(A.shape[1]):
            A[:,j] /= max(np.linalg.norm(A[:,j]),1e-20)

        #print('Antes de entrar a omp el diccionario tiene la forma: ', A.shape)
        #print('Antes de entrar a omp los datos objetivo: ', Y.shape)
        omp.fit(A,Y)
        return omp.coef_.T

    def _update_parameters(self,Y,A,X):
        for j in range(self.rank):
            #print(' >>> Iteraciones X tiene la forma: ',X.shape)
            NZ = np.where(X[j, :] != 0)[0]
            A_tmp = A
            X_tmp = X
            if len(NZ) > 0:
                A_tmp[:,j][:] = 0
                E_R = Y[:,NZ]-A_tmp.dot(X_tmp[:,NZ])

                if self.func_svd is True:
                    u, s, v = np.linalg.svd(E_R)
                    X[j,NZ] = s[0]*np.asarray(v)[0]
                    A[:,j] = u[:,0]
                else:
                    A_R = E_R.dot(X[j,NZ].T)
                    A_R /= np.linalg.norm(A_R)
                    X_R = E_R.T.dot(A_R)
                    X[j,NZ] = X_R.T
                    A[:,j] = A_R

        return A_tmp,X_tmp

    def _edit_dictionary(self,Y,A,X):

        E = Y-A.dot(X)
        E_atom_norm = np.linalg.norm(E,axis=0).tolist()
        Max_index = E_atom_norm.index(max(E_atom_norm))
        examp = np.matrix(Y[:,Max_index])

        for j in range(A.shape[1]):
            for k in range(j+1,A.shape[1]):
                if np.linalg.norm(A[:,j]-A[:,k]) < 1e-1:
                    A[:,k]=examp+np.random.randn(1,Y.shape[0])*0.0001

        for j in range(X.shape[0]):
            if np.linalg.norm(X[j,:])/X.shape[1] < 1e-5:
                A[:,j]=examp+np.random.randn(1,Y.shape[0])*0.0001

        return A

    def fit(self, Y): #recibo la forma de las matrices
        """
        Y = AX      #Diccionario X Vector de dispersion
        Y: shape = [n_features,n_samples]
        A: Dictionary = [n_features, rank]
        X: Sparse = [rank, n_samples]
        """
        err = 1e+8
        err_f = 1e+10
        registro_errores = []
        A, X = self._initialize_parameters(Y)
        for j in range(self.max_iter):

            X_tmp = self._estimate_X(Y,A)
            X_temp = np.reshape(X_tmp, (X_tmp.shape[0],1))
            plt.plot(A.dot(X_tmp), label='OMP iteration: '+str(j))
            plt.legend(loc='best')
            err = np.linalg.norm(Y-A.dot(X_tmp))
            registro_errores.append(err)
            print('Iteracion: ', j,':error=',err/(Y.shape[0]*Y.shape[1]))
            if err < self.max_tol:
                break
            if err < err_f :
                err_f = err
                X = X_tmp
                X = np.reshape(X, (X.shape[0],1))
            A,X = self._update_parameters(Y,A,X)
            A = self._edit_dictionary(Y,A,X)
        plt.show()
        plt.plot(registro_errores, 'o-', color='red')
        plt.title('Convergencia de error')
        plt.ylabel('Error')
        plt.xlabel('Iteraciones')
        plt.show()
        return A, X
