import pandas as pd


import numpy as np
from itertools import combinations
from scipy.special import comb
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import linear_model
data = pd.read_excel('Data.xlsx')
day_return = (np.array(data.iloc[1:, 0:7])-np.array(data.iloc[0:-1,0:7]))/np.array(data.iloc[0:-1, 0:7])
day_return = pd.DataFrame(day_return, columns=data.columns, index=data.index[1:])
T = day_return.shape[0]
S = 14
sub_data = {}
for i in range(S):
    sub_data[str(i+1)] = day_return.iloc[int(T/S*i):int(T/S*(i+1)),]
S_total = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
str(range(14))
lambdac = []
sris = []
sroos = []

for i1, i2, i3, i4, i5, i6, i7 in combinations(S_total, 7):
    train = list(combinations(S_total, 7))[0]
    train = [i1, i2, i3, i4, i5, i6, i7]
    test = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    trainset = pd.DataFrame(columns=day_return.columns)
    testset = pd.DataFrame(columns=day_return.columns)
    for i in train:
        trainset = trainset.append(sub_data[i])
        test.remove(i)
    for j in test:
        testset = testset.append(sub_data[j])
    trainR = trainset.mean()/trainset.std()
    testR = testset.mean()/testset.std()
    sris.append(max(trainR))
    nstar = np.where(trainR == max(trainR))
    wc = sorted(testR).index(testR.iloc[nstar].tolist()[0])
    sroos.append(testR.iloc[nstar].tolist()[0])
    wc = (wc+1)/8
    lambdac.append(np.log(wc/(1-wc)))
#计算pbo
pbo = 0
for i in lambdac:
    if i < 0:
        pbo = pbo+1
pbo/len(lambdac)

#sris 和 sroos
rger.coef_
rger.intercept_
rger = linear_model.LinearRegression()
rger.fit(np.array(sris).reshape(-1, 1), sroos)
pre_rsoos = rger.predict(np.array(sris).reshape(-1, 1))
plt.figure(figsize=(10, 8))
plt.plot(sris, pre_rsoos, linewidth=4, color='red')
plt.plot(sris, sroos, '.', color='blue')
plt.xlabel('SR IS')
plt.ylabel('SR OOS')
plt.title('Best Sharpe ratio in-sample (SR IS) vs Sharpe ratio out-of-sample (SR OOS)')

#lambdac分布直方图
plt.figure(figsize=(10, 5))
sns.distplot(lambdac, kde=True, bins=10, rug=True)
plt.ylabel('frequency')
plt.xlabel('logits')
plt.title('Hist of Rank logits')
plt.show()
