#%%
import sqlite3 # sqlite3 데이터베이스
import re      # 정규식
import numpy as np # 숫자 라이브러리    
import pandas as pd # 데이터처리 라이브러리
import matplotlib.pyplot as plt # 그래프 라이브러리
import matplotlib
import seaborn as sns # 그래프 고도화
# scikit-learn

# %%
from sklearn import datasets as data
iris = data.load_iris()
#전처리
irdata=iris.data
irtgt=iris.target
feature=iris.feature_names
tgtname=iris.target_names
# %%
import pandas as pd
feature=['sl','sw','pl','pw']
df = pd.DataFrame(irdata, columns= feature)
df.plot(style='.') # 기초 그래프
# %%
df.describe() #기초통계량
# %%
df.info() # 데이터타입 요약
# %%
plt.plot(irtgt,'.')
# %%
plt.hist(irtgt)
# %%
df['tgt'] = irtgt;df
# %%
sns.pairplot(df, hue='tgt', palette='winter')
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test  = train_test_split(irdata, irtgt, test_size=0.3, shuffle=True, random_state=1)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)
# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# 하이퍼파라미터 튜닝
for i in range(3,20,2):
    print('knn:',i)
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(X_train,Y_train)
    pred=knn3.predict(X_test)
    print(pred)
    print(Y_test)
    acc=accuracy_score(pred,Y_test)
    print('점수[',i,']', acc)

# %%
from sklearn.svm import SVC
for i in range(1,10):
    svc= SVC(C=i)
    svc.fit(X_train,Y_train)
    pred=svc.predict(X_test)
    print(pred)
    print(Y_test)
    acc=accuracy_score(pred,Y_test)
    print('svc[',i,'] acc', acc)

# %%
from sklearn.tree import DecisionTreeClassifier as DT
for j in range(2,10):
    for i in range(2,10):
        dt= DT(max_depth=i, min_samples_leaf=j)
        dt.fit(X_train,Y_train)
        pred=dt.predict(X_test)
        print(pred)
        print(Y_test)
        acc=accuracy_score(pred,Y_test)
        print('dt[',i,',',j,'] acc', acc)

# %%
from sklearn.ensemble import RandomForestClassifier as RF
rf = RF()
rf.fit(X_train,Y_train)
pred=rf.predict(X_test)
print(pred)
print(Y_test)
acc=accuracy_score(pred,Y_test)
print('rf acc', acc)
