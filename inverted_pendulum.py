'''
=====================================
Author  :  Muhan Zhao
Date    :  Feb. 11, 2020
Location:  UC San Diego, La Jolla, CA
=====================================
'''
import numpy as np
import sympy as sp
from define_system import ControlAffineSystem


class InvertedPendulum:
    '''
    Define the symbolic dynamic:    dx = f(x) + g(x) * u
    x contains two states:  x1 -> theta     x2 -> theta-dot


    '''
    def __init__(self, l, m, g, b=0):
        self.l = l
        self.m = m
        self.g = g
        self.b = b
        x1, x2 = sp.symbols('x1 x2')
        self.x = [x1, x2]
        self.f, self.g = self.inverted_pendulum_dynamics()

    def inverted_pendulum_dynamics(self):

        f = sp.Matrix([[self.x[2]],
                      [-self.b*self.x[2] + (self.m*self.g*self.l*np.sin(self.x[0])/2)/(self.m*self.l**2/3)]])
        g = sp.Matrix([[0], [-1/(self.m*self.l**2/3)]])
        return f, g




