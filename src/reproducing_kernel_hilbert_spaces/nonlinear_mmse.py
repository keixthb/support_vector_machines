import numpy as np 
import scipy.signal as sp 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from numpy.linalg import inv
import math
import os

def print_shape(z):
  rows,columns = z.shape
  print("Length (number of rows):", rows)
  print("Width (number of columns):", columns)

def data(N,a):
   N=N+3
   h=np.array([1,a])
   y=np.sign(np.random.randn(N,1))
   y=np.reshape(y,len(y))
   t=np.arange(N)
   z=np.convolve(h,y)
   z=z[1:N-1]
   y=y[2:N-1]
   X=np.array([z[0:N-3],z[1:N-2]])+0.2*np.random.randn(2,N-3)
   return X,y

def volterra(X):
  phi = np.zeros((10, X.shape[1]))
  for i in range(X.shape[1]):
      x_n_minus_1 = X[0, i]
      x_n = X[1, i]

      phi[:, i] = [1,
                   x_n,
                   x_n_minus_1,
                   x_n**2,
                   x_n_minus_1**2,
                   x_n * x_n_minus_1,
                   x_n**3,
                   x_n_minus_1**3,
                   x_n**2 * x_n_minus_1,
                   x_n * x_n_minus_1**2]
  return phi

def plot_decision_function(classifier, X, y,  limits=(-4,4), title=""):
    xx, yy = np.meshgrid(np.linspace(limits[0], limits[1], 500), np.linspace(-4, 4, 500))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.contour(xx, yy, Z, cmap='gray',levels=0)

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        alpha=1,
        cmap="gray",
        edgecolors="black",
    )

    plt.ylabel(r'$x_n$',fontsize=14)
    plt.xlabel(r'${x_{n-1}}$',fontsize=14)
    plt.title(title)
    index = 0
    while True:
        file_name = f"fig_{index}.png"
        if not os.path.exists(file_name):
            plt.savefig(file_name)
            print(f"Figure saved as: {file_name}")
            break
        else:
            index += 1
    plt.legend()
    plt.show()
    plt.savefig("fig2.png")
    plt.cla()


def plot_data(X, y, alpha):
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(X[0, :], X[1, :], c=y, alpha=1, cmap="gray", edgecolors="black")
    plt.ylabel(r'$x_n$',fontsize=14)
    plt.xlabel(r'${x_{n-1}}$',fontsize=14)
    plt.title(f'Training Data $\\alpha$ = {alpha}')
    index = 0
    while True:
        file_name = f"fig_{index}.png"
        if not os.path.exists(file_name):
            plt.savefig(file_name)
            print(f"Figure saved as: {file_name}")
            break
        else:
            index += 1
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend()
    plt.show()
    plt.cla()
    return

def plot_decision_function_voltera(classifier, X, y,  limits=(-4,4), title=""):
    xx, yy = np.meshgrid(np.linspace(limits[0], limits[1], 500), np.linspace(-4, 4, 500))
    #Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    full = np.vstack((xx.ravel(), yy.ravel()))
    tempPhi = volterra(full)

    Z = classifier.predict(tempPhi.T)
    print(len(Z))
    Z = Z.reshape(xx.shape)
    print_shape(Z)
    plt.contour(xx, yy, Z, cmap='gray',levels=0)

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        alpha=1,
        cmap="gray",
        edgecolors="black",
    )

    plt.ylabel(r'$x_n$',fontsize=14)
    plt.xlabel(r'${x_{n-1}}$',fontsize=14)
    plt.title(title)

    index = 0
    while True:
        file_name = f"fig_{index}.png"
        if not os.path.exists(file_name):
            plt.savefig(file_name)
            print(f"Figure saved as: {file_name}")
            break
        else:
            index += 1
    plt.show()
    plt.cla()

class ridge:
  def __init__(self,gamma=0.001): 
    self.gamma_ = gamma

  def fit(self, X, y, gamma=-1):
    if gamma != -1:
      self.gamma_ = gamma  
                          
    d=X.shape[1]
    self.w_= inv(X.T@X+self.gamma_*np.identity(d))@X.T@y 

  def predict(self,X):
    y_=X@self.w_   
    return y_

  def crossval(self, X, y): 
    Nint=20
    T=np.log10(np.trace(X.T@X))
    gamma_int=np.logspace(T-3, T-2,Nint)
    N=X.shape[0] 
    E=np.zeros(Nint) 

    for j in range(Nint): 
      for i in range(N):  
        Xtr=np.concatenate((X[:i,:],X[i+1:,:]))
        Xval=X[i,:]
        ytr=np.concatenate((y[:i],y[i+1:]))
        yval=y[i]

        self.gamma_=gamma_int[j]

        self.fit(Xtr, ytr)

        E[j]=E[j]+(yval-self.predict(Xval))**2 

    self.gamma_=gamma_int[np.argmin(E)] 
    self.fit(X, y) 

spread = 1.5


X,y=data(100,spread)


plot_data(X, y, spread)


phi = volterra(X)
phi_t = phi.T

clf = ridge()
clf.crossval(phi_t,y) 

plot_decision_function_voltera(classifier=clf, limits=(-4,4), title=f"Volterra Expansion $\\alpha$ = {spread}", X=X.T, y=y)