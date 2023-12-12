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






nu_size = 100
C_size = 100
Nu = np.logspace(-3,0,nu_size)
C = np.logspace(-2,2,C_size)
ErrorsNu = np.zeros((C.size,Nu.size))
leastError = 1
optimalNu = 0
optimalC_N = 0

for i in range(C.size):
  for j in range(Nu.size):
    nu_model = NuSVR(kernel='linear',nu=Nu[j], C=C[i])
    nu_model.fit(Xtrain80,Ytrain80)
    prediction = nu_model.predict(Xvalid)


    ErrorsNu[i,j] = mean_squared_error(Yvalid, prediction)
    if (leastError > ErrorsNu[i,j]):
      leastError = ErrorsNu[i,j]
      optimalNu = Nu[j]
      optimalC_N = C[i]

nu_SVR = NuSVR(kernel='linear',nu=0.5)
nu_SVR.fit(Xtrain,Ytrain)
predictionNu = nu_SVR.predict(Xtest)


plt.figure(figsize=(19, 6.5))
plot = plt.contourf(ErrorsNu)

plt.xlabel('c')
plt.xticks(range(len(C)), C)
xticks = plt.xticks()[0]
new_xticks = xticks[::tickRate(C_size, 3)]
plt.xticks(new_xticks)

plt.ylabel('$\\nu$')
plt.yticks(range(len(Nu)), Nu)
yticks = plt.yticks()[0]
new_yticks = yticks[::tickRate(nu_size, 10)]
plt.yticks(new_yticks)

plt.title('validation error')
plt.colorbar(plot)

plt.savefig("nu_validation_error.png")
plt.show()
plt.cla()



plt.figure(figsize=(19, 6.5))
plt.plot(Ytest, color='orange', label='validation')
plt.plot(predictionNu, color='purple', linestyle='--', label='testing')

plt.xlabel('actual')
plt.ylabel('predicted')
plt.title(f'$\\nu$ -svr $\\nu$ = {optimalNu:.6} , c = {optimalC_N:.3}')
plt.legend()

plt.savefig("nu_results.png")
plt.show()
