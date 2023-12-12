import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.svm import SVR, NuSVR
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FuncFormatter

def intervalOut(start,end,interval):
  arr = np.arange(start,end + interval,interval)
  return arr

def tickRate(size,rate):
  return int(size / rate)

fullCSV = pd.read_csv('data.csv',header=None)

ftest = fullCSV.iloc[:,0]
ftrain = fullCSV.iloc[:,1]

Xtest = fullCSV.iloc[:,2:20]
Xtrain = fullCSV.iloc[:,21:39]

Ytest = fullCSV.iloc[:,40]
Ytrain = fullCSV.iloc[:,41]

Xtrain80 = fullCSV.iloc[0:63:,21:39]
Xvalid = fullCSV.iloc[64:80:,21:39]
Ytrain80 = fullCSV.iloc[0:63:,41]
Yvalid = fullCSV.iloc[64:80,41]

eps_size = 100
C_size = 100
Eps = np.logspace(-4,-1,eps_size)
C = np.logspace(-2,2,C_size)
ErrorsEps = np.zeros((C.size,Eps.size))
leastError = 1
optimalEps = 0
optimalC_E = 0

for i in range(C.size):
  for j in range(Eps.size):
    ep_model = SVR(kernel='linear', epsilon=Eps[j], C=C[i])
    ep_model.fit(Xtrain80,Ytrain80)
    prediction = ep_model.predict(Xvalid)

    ErrorsEps[i,j] = mean_squared_error(Yvalid, prediction)
    if (leastError > ErrorsEps[i,j]):
       leastError = ErrorsEps[i,j]
       optimalEps = Eps[j]
       optimalC_E = C[i]



epsilon_SVR = SVR(kernel='linear', epsilon=optimalEps)
epsilon_SVR.fit(Xtrain,Ytrain)
predictionE = epsilon_SVR.predict(Xtest)



plt.figure(figsize=(19, 6.5))
plot = plt.contourf(ErrorsEps)

plt.xlabel('c')
plt.xticks(range(len(C)), C)
xticks = plt.xticks()[0]
new_xticks = xticks[::tickRate(C_size, 3)]
plt.xticks(new_xticks)

plt.ylabel('$\epsilon$')
plt.yticks(range(len(Eps)), Eps)
yticks = plt.yticks()[0]
new_yticks = yticks[::tickRate(eps_size, 10)]
plt.yticks(new_yticks)

plt.title('validation error')
plt.colorbar(plot)


plt.savefig("epsilon_validation_error.png")
plt.show()
plt.cla()




plt.figure(figsize=(19, 6.5))
plt.plot(Ytest, color='orange', label='validation')
plt.plot(predictionE, color='purple', linestyle='--', label='testing')

plt.xlabel('actual')
plt.ylabel('predicted')
plt.title(f'$\epsilon$-svr $\epsilon$ = {optimalEps:.6} , c = {optimalC_E:.3}')
plt.legend()

plt.savefig("epsilon_results.png")
plt.show()
