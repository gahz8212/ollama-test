from __future__ import annotations
#어떤 것을 처리하기 위한 부품을 만들어버리자
# 여러 부품을 만들때 틀을 먼저 만들어두고
# 부품이 필요하면 틀을 찍어서 사용할 대상(부품)을 만든다.
# Car 틀을 먼저 만든다.
# --> 내 차가 필요하면 Car 틀을 이용해서 차를 만들고
# --> 내 차의 특징을 상세하게 넣어 줌.
# 만들 대상(부품, objhect)
# --> 대상으로(Object, 객체)로 만들어서 코딩하는 방식


class Car: #차에 대한 틀을 만들자 -->  일반적인 특징으로 만들자!
    # 특징(속성)
    color: str="검정색"
    price: float
    def __init__(self,color:str,price:float):
        self.color=color
        self.price=price
    def __str__(self):
        return(f"{self.color} {self.price}")
    # 특징(동작)
    def run(self):
        print('달리다.')
    def speed(self):
        print('속도를 올리다.')

    @staticmethod
    def start():
        print('만든 회사이름은 현대자동차이다.')

if __name__ == '__main__':
    mycar = Car(color="red",price=100)
    yourcar=Car(color="yellow",price=200)

    print(mycar)
    mycar.run()
    mycar.speed()
    Car.start()

