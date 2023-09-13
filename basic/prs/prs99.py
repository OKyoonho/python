def make99():
    for i in range(2,10):
        print("")
        for j in range(1,10):
            ans = i*j
            if((i*j)%2 == 1):
                print(ans,'*', end="\t")
            else:
                print(ans, end="\t")
class person():
    def __init__(self,name,age):
        self.name = name #클래스 속성
        self.age = age 
    def sayhello(self): #클래스의 메서드이다 self
        print('hello '+self.name)
        print("I'm "+self.age+' old')
