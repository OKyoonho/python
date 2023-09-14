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
cancer = load_breast_cancer()
# %%
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['tgt']= cancer.target;df
# %%
from pycaret.classification import *
cres = setup(df,target='tgt', train_size = 0.8, session_id=0);cres
# %%
bestmodel = compare_models(sort='Accuracy')
# %%
bestmodel
# %%
pred=predict_model(bestmodel)
pred

# %%
fmodel=finalize_model(bestmodel)
fmodel
# %%
pred=predict_model(bestmodel, data=df.iloc[:100])
pmean=pred['prediction_score'].mean()
pmean
# %%
bestmodel = compare_models(sort='Accuracy')
bestmodel
# %%
