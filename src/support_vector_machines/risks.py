import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from sklearn import svm
import time


####################################################################################################
#data functions and error calculation provided
#by Dr. Manel Martinez-Ramon
####################################################################################################

def data(N, sigma):
    w = np.ones(10) / np.sqrt(10)
    w1 = [1., 1., 1., 1., 1., -1., -1., -1., -1., -1.] / np.sqrt(10)
    w2 = [-1., -1., 0., 1., 1., -1., -1., 0., -1., -1.] / np.sqrt(8)
    x = np.zeros((4,10))
    x[1, :] = x[0 ,:] + sigma * w1
    x[2, :] = x[0 ,:] + sigma * w2
    x[3, :] = x[2 ,:] + sigma * w1
    X1 = x + sigma * matlib.repmat(w, 4, 1) / 2
    X2 = x - sigma * matlib.repmat(w, 4, 1) / 2
    X1 = matlib.repmat(X1, 2 * N, 1)
    X2 = matlib.repmat(X2, 2 * N, 1)
    X = np.concatenate((X1, X2), axis = 0)
    Y = np.concatenate((np.ones(4 * 2 * N), -np.ones(4 * 2 * N)), axis = 0)
    Z = np.random.permutation(16 * N)
    Z = Z[:N]
    X = X[Z,:]
    X = X + 0.2 * sigma * np.random.randn(N,10)
    Y = Y[Z]
    return X, Y

def data2(Ntr, Ntst, sigma):
    Xtr, ytr = data(Ntr, sigma)
    Xtst, ytst = data(Ntst, sigma)
    return Xtr, ytr, Xtst, ytst

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

axises = [ax1, ax2, ax3]

def main()->None:
    T = 100
    C = np.logspace(0.01, 10, 20)
    Etr = np.zeros((C.size, T))
    Etst = np.zeros((C.size, T))

    standard_deviation_of_the_clusters = [0.5, 1, 3]
    for sigma, axis in zip(standard_deviation_of_the_clusters, axises):
        start_time = time.time()
        for j in range(T):
            print(f"iteration:\t{j}, sigma:\t{sigma}")
            Xtr, ytr, Xtst, ytst = data2(100, 100, sigma)
            for i in range(C.size):
                clf = svm.SVC(kernel = "linear", C = C[i])
                clf.fit(Xtr, ytr)
                ytr_ = clf.predict(Xtr)
                ytst_ = clf.predict(Xtst)
                Etr[i, j] = np.mean(abs(ytr - ytr_) / 2)
                Etst[i, j] = np.mean(abs(ytst - ytst_) / 2)

        R_emp = np.mean(Etr, axis = 1)
        R = np.mean(Etst, axis = 1)


        axis.semilogx(C, R_emp, label = "emperical_risk")
        axis.semilogx(C, R, label = "risk")
        axis.semilogx(C, R - R_emp, label = "risk - emperical_risk")
        axis.set_xlabel('values of c')
        axis.set_ylabel('average error')
        axis.set_title(f"sigma = {sigma}")
        axis.legend()

        elapsed_time = time.time() - start_time
        print(f'run time for sigma:\t{sigma}\t{elapsed_time} seconds\n')




    fig.suptitle('Support Vector Machine')

    plt.savefig("risks.png")
    plt.show()


    plt.cla()



    return




if("__main__" == __name__):
    main()
