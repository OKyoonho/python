#%%
print('hello world')

#%%
#단열 주석
''' 문자열이지만 다열주석으로 사용가능'''
a=3
b=5
c=a+b
print(c)
print('나머지연산', b%a)#나머지
b//a #몫연산  print 생략해도 주피터상에서는 출력가능

#%%
#데이터셋: array
    #리스트: [] - 일반배열
    #튜플: () - 수정 불가능한 리스트
    #딕셔너리: {키:밸류} - 연관배열: 객체의 개념
    #셋: set() 집합

li = [1,2,"3",[4,5]]
li[0] = 0
print('0치환후:', li)
# %%
tu = (1,2,'3',[4,5])
tu[0]=0
print('0치환후:', tu)
# %%
di={'이름':'홍길동',1:19,2:[1,2,3]}
print(di[2])
# %%
for i in li:
    print(i)
# %%
# dic은 key값만 빠짐
for d in di:
    print(d)
print("#"*30)
for k, v in di.items():
    print(k,":",v)

# %%
#구구단 만들기
for i in range(2,10):
    print("")
    for j in range(1,10):
        ans = i*j
        if((i*j)%2 == 1):
            print(ans,'*', end="\t")
        else:
            print(ans, end="\t")

# %%
class person():
    def __init__(self,name,age):
        self.name = name #클래스 속성
        self.age = age 
    def sayhello(self): #클래스의 메서드이다 self
        print('hello '+self.name)
        print("I'm "+self.age+' old')
anna = person('anna', '19')
anna.sayhello()

# %%
