import numpy as np 
import scipy.signal as sp 
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
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

class dual:
  def __init__(self,gamma,pwr): 
    self.gamma_ = gamma
    self.power_ = pwr

  def fit(self, X, y):
    self.X_train = X
    x_prod = np.dot(X,X.T)
    lx = x_prod.shape[0]

    
    K = (x_prod + self.gamma_*np.ones((lx,lx)))**self.power_

    
    temp = np.linalg.inv(K + self.gamma_*np.identity(lx))
    self.alpha_ = temp@y

  def predict(self, X):
    x_prod = np.dot(X,self.X_train.T)
    l_row, l_col = x_prod.shape

    
    K_tst = (x_prod + self.gamma_*np.ones((l_row,l_col)))**self.power_
    y = K_tst@self.alpha_ 
    return y

#spread = 0.2
spread = 1.5

X, y = data(100, spread)

obj = dual(0.0075,3)
obj.fit(X.T,y)
plot_data(X, y, spread)
plot_decision_function(classifier=obj, limits=(-4,4), title=f"dual kernel method $\\alpha$={spread}", X=X.T, y=y)