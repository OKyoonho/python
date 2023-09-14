#%%
import sqlite3 # sqlite3 데이터베이스
import re      # 정규식
import numpy as np # 숫자 라이브러리    
import pandas as pd # 데이터처리 라이브러리
import matplotlib.pyplot as plt # 그래프 라이브러리
import matplotlib
import seaborn as sns # 그래프 고도화
# %%
from konlpy.tag import Kkma
kkma = Kkma()
# %%
res = kkma.pos('안녕하세요 그런데 여러분 만나서 반갑습니다.')
res
# %%
def getPOS(txt='안녕하세요 그런데 여러분 만나서 반갑습니다.'):
    res = kkma.pos(txt)
    # reqPos=['NNG','NNP','NP','VV','VA','VCP','VCN','JC','MAC','EFN','EFA','EFQ','EFO','EFA','EFI','EFR']
    reqPos=['NNG','NNP','NP','VV','VA','VCN','JC','MAC','EFA','EFQ','EFO','EFA','EFI','EFR']#,'VCP','EFN'
    wset=[]
    for r in res:
        if(r[1] in reqPos):
            wset.append(r[0])
            # print(r)
    return(' '.join(wset))
getPOS()
# %%
fname = './src/현진건-운수_좋은_날+B3356-개벽.txt'
with open(fname, encoding='utf-8') as f:
    r = f.readlines()
lucky = ''.join(r)
# %%
lucky = lucky.replace('\n\n', '{nn}')
lucky = lucky.replace('\n', '')
lucky = lucky.replace('{nn}','.')
lucky = lucky.split('.')
lucky
# %%
copus=[]
for luck in lucky:
    ltxt = getPOS(luck)
    copus.append(ltxt)
# %%
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cvect = CountVectorizer()
cvfit=cvect.fit_transform(copus)
cvtable = cvfit.toarray()
print(cvtable.shape)
print(cvect.vocabulary_)
# %%
plt.imshow(cvtable[:100,:100])
# %%
tvect = TfidfVectorizer()
tvfit=tvect.fit_transform(copus)
tvtable = tvfit.toarray()
print(tvtable.shape)
print(tvect.vocabulary_)
plt.imshow(tvtable[:100,:100])
# %%
cdf=pd.DataFrame(cvtable[:100,:100])
cdf
# %%
tdf=pd.DataFrame(tvtable[:100,:100])
tdf

# %%
