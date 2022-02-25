import numpy as np

class Quadratic():

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def discr(self):
        return self.b**2  - 4*self.a*self.c

    def solve(self, d):

        if d >= 0: 
            sol = ((self.b - np.sqrt(d))/(2* self.a), (- self.b - np.sqrt(d))/(2*self.a))
            return sol

a = 2   
b = np.random.uniform(-3.0, 4.0)
c = np.random.uniform(-3.0, 4.0)

Q = Quadratic(a, b, c)
d = Q.discr()
print(Q.solve(d))