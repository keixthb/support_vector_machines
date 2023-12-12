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

allGammas = np.logspace(-2,1,1000)
allMSE = []
optimalGamma = 0
minMSE = 0
itr = 0

for i in allGammas:
    lr_model = Ridge(alpha=i)
    lr_model.fit(Xtrain80,Ytrain80)
    predictions = lr_model.predict(Xvalid)

    mse = mean_squared_error(Yvalid, predictions)
    allMSE.append(mse)

    if itr == 0:
      itr += 1
      minMSE = mse

    if minMSE > mse:
      minMSE = mse
      optimalGamma = i

lr_model = Ridge(alpha=optimalGamma).fit(Xtrain,Ytrain)
predictions = lr_model.predict(Xtest)



plt.figure(figsize=(15, 6.5))
plt.plot(allGammas, allMSE)
plt.xlabel('$\gamma$')
plt.xscale('log')
plt.ylabel('error')
plt.ylim(0.009, 0.027)
plt.title('Validation Squared Error')
plt.savefig("linear_validation_error.png")
plt.show()
plt.cla()



plt.plot(Ytest, color='orange', label='validation')
plt.plot(predictions, color='purple', linestyle='--', label='testing')

plt.xlabel('actual')
plt.ylabel('predicted')
plt.title(f'$\gamma$ = {optimalGamma:.6}')
plt.legend()

plt.savefig("linear_results.png")
plt.show()
