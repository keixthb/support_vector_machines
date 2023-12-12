import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

def samples(total,sigma):
  X = np.random.rand(total,1)
  gamma_sqr = np.random.normal(0,sigma,size=(total,1))
  y = 0.5*X + 0.5 + gamma_sqr
  return X,y

sigma = 0.05
total_samples = 50
test_samples = 10

X,y = samples(total_samples,sigma)
hilbert_space = np.linspace(0,1,test_samples).reshape(-1,1)

plt.scatter(X, y, c='blue', label='Data Points')
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title(f"Data Sample Chart | σ = {sigma}")
plt.savefig("one.png")
plt.show()

kernel = 1*DotProduct(0.1) + WhiteKernel(0.1)
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X, y)
gaussian_process.kernel

Y_predict, Sig_predict = gaussian_process.predict(hilbert_space, return_std=True)
plt.scatter(X, y, c='blue', label='Data Points')
plt.plot(hilbert_space, Y_predict, color="green")
plt.scatter(hilbert_space, Y_predict, color="red")

plt.plot(hilbert_space, Y_predict - 1 * Sig_predict, linestyle=(0,(1,1)), color="black", label="One band")
plt.plot(hilbert_space, Y_predict + 1 * Sig_predict, linestyle=(0,(1,1)), color="black")
plt.plot(hilbert_space, Y_predict - 2 * Sig_predict, linestyle=(0,(1,3)), color="black", label="Two band")
plt.plot(hilbert_space, Y_predict + 2 * Sig_predict, linestyle=(0,(1,3)), color="black")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(f"One Dimensional Regression")
plt.show()
plt.savefig("two.png")
