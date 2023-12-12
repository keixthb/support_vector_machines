import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel


def sliding_window(x, w = 1, d = 0):
    N = len(x)
    m = int(np.ceil((N-w)/(w-d)) + 1)
    X = np.zeros([w,m])
    for i,j in zip(range(0,N,w-d),range(0,m)):
        X[:,j] = x[i:i + w]
    return X

def buffer(x, n, p=0):
    i = 0
    result = x[:n]
    i = n
    result = list(np.expand_dims(result, axis=0))
    while i < len(x):
        col = x[i:i+(n-p)]
        if p != 0:
            col = np.hstack([result[-1][-p:], col])
        if len(col):
            col = np.hstack([col, np.zeros(n - len(col))])
        result.append(np.array(col))
        i += (n - p)
    return np.vstack(result).T

def data(N, m, sigma):
    m = m+1
    np.random.seed(100)
    x=np.random.randn(1, N)
    b, a = signal.butter(4, 0.05)
    f = signal.filtfilt(b, a, x)
    temp = f + sigma*np.random.randn(1, N)
    temp = buffer(temp, m, m-1)
    y = temp[-1,:].T
    Y = temp[0:-2,:]
    return Y, y

N_train=40
W=3
M=3

Y,y=data(100,1,0)
Y_noise,y_noise=data(100,1,0.05)
new_hilbert_space = np.linspace(0,1,98).reshape(-1,1)

plt.figure(figsize=(8,6))

plt.scatter(new_hilbert_space[N_train+M+W-1:],Y_noise[N_train+M+W-1:],marker='o', facecolors='none',color='red', label='Real')
plt.plot(new_hilbert_space[N_train+M+W-1:],Y[N_train+M+W-1:],color='red', label='Noise Free Output')

x_train=Y_noise[:N_train].flatten()
t_train=new_hilbert_space[M+W-1:N_train].flatten()
x_test=Y_noise[N_train:].flatten()
t_test=new_hilbert_space[N_train+M+W-1:].flatten()

X_train=sliding_window(x_train[:-M],W,W-1).T
y_train=x_train[M+W-1:]

X_test=sliding_window(x_test[:-M],W,W-1).T
y_test=x_test[M+W-1:]

kernel = 1*DotProduct(0.1) + WhiteKernel(0.1)
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)

print(f"t_test shape{t_test.shape}")
print(f"y_test shape{y_test.shape}")
print(f"mean_prediction shape{mean_prediction.shape}")

plt.plot(t_test, mean_prediction, label="Predicted",color="black")
plt.plot(new_hilbert_space[N_train+M+W-1:], mean_prediction - 1.96 * std_prediction, linestyle=(0,(1,1)), color="black", label="One $\sigma$ Interval")
plt.plot(new_hilbert_space[N_train+M+W-1:], mean_prediction + 1.96 * std_prediction, linestyle=(0,(1,1)), color="black")

plt.legend()
plt.xlabel("$t$")
plt.ylabel("$f(t)$")
plt.title("Gaussian process regression")
plt.savefig("three.png")
plt.show()
plt.cla()








plt.figure(figsize=(8,6))

plt.scatter(new_hilbert_space[N_train+M+W-1:],Y_noise[N_train+M+W-1:],marker='o', facecolors='none',color='red', label='Real')
plt.plot(new_hilbert_space[N_train+M+W-1:],Y[N_train+M+W-1:],color='red', label='Noise Free Output')



plt.legend()
plt.xlabel("$t$")
plt.ylabel("$f(t)$")
plt.title("Data")
plt.savefig("four.png")
plt.show()
