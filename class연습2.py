from __future__ import annotations

class Brick:

    def __init__(self,size,color):
        self.color=color
        self.size=size

    def stack_bricks(self):
        self.size+=1
    def __str__(self):
        return self.color+' brick is '+str(self.size)+'size'

class Hammer:
    @staticmethod
    def hit_with_hammer():
        print('벽돌을 두드리다.')

if __name__ == '__main__':
    brick1=Brick(size=100,color="Red")
    brick2=Brick(size=200,color="Blue")
    print(brick1)
    print(brick2)
    Hammer.hit_with_hammer()
