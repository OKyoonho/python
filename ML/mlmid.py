#%%
import sqlite3 # sqlite3 데이터베이스
import re      # 정규식
import numpy as np # 숫자 라이브러리    
import pandas as pd # 데이터처리 라이브러리
import matplotlib.pyplot as plt # 그래프 라이브러리
import matplotlib
import seaborn as sns # 그래프 고도화
# %%
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data['data']
Y = data['target']
fname = data['feature_names']
tname = data['target_names']
df = pd.DataFrame(X, columns=fname);df
# %%
df.describe()
df.info()
plt.hist(Y)
#%%
plt.plot(df['mean radius'],'.')
sns.scatterplot(df.iloc[:,:3])
#%%
tdf = df.copy()
tdf['tgt']=Y
sns.pairplot(tdf.iloc[:,-5:], hue='tgt')
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, shuffle=True, random_state=1, stratify=Y)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
plt.hist(Y_train)
plt.hist(Y_test)

#%%
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
def makeRF(i,j):
    rf = RF(max_depth=j, max_leaf_nodes=i)
    rf.fit(X_train,Y_train)
    pred=rf.predict(X_test)
    # print(pred)
    # print(Y_test)
    acc=accuracy_score(pred,Y_test)
    # print('rf acc', acc)
    return acc
accli=[]
beforeACC=0
bestACC = []
for i in range(2,10):
    for j in range(2,10):
        acc = makeRF(i,j)
        if (acc>beforeACC):
            bestACC = [i,j,acc]
        beforeACC = acc
        accli.append(acc)
        
#%%
print(bestACC)
plt.plot(accli)

# %%
from sklearn.ensemble import GradientBoostingClassifier as GB
def makeGB(i,j):
    gb = GB(min_samples_split=j, n_estimators=i*50)
    gb.fit(X_train,Y_train)
    pred=gb.predict(X_test)
    # print(pred)
    # print(Y_test)
    acc=accuracy_score(pred,Y_test)
    print('GB[',i,',',j,']', acc)
    return acc
accli=[]
beforeACC=0
bestACC = []
for i in range(1,10):
    for j in range(2,10):
        acc = makeGB(i,j)
        if (acc>beforeACC):
            bestACC = [i,j,acc]
        beforeACC = acc
        accli.append(acc)
plt.plot(accli)
# %%
from sklearn.ensemble import AdaBoostClassifier as AB
def makeAB(i):
    ab = AB(n_estimators=i*50)
    ab.fit(X_train,Y_train)
    pred=ab.predict(X_test)
    # print(pred)
    # print(Y_test)
    acc=accuracy_score(pred,Y_test)
    print('AB[',i,',',j,']', acc)
    return acc
accli=[]
beforeACC=0
bestACC = []
for i in range(1,10):
    acc = makeAB(i)
    if (acc>beforeACC):
        bestACC = [i,acc]
    beforeACC = acc
    accli.append(acc)
plt.plot(accli)
#%%
