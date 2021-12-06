'''
=====================================
Author  :  Muhan Zhao
Date    :  Feb. 11, 2020
Location:  UC San Diego, La Jolla, CA
=====================================
'''
import numpy as np
import sympy as sp
# from define_system import ControlAffineSystem


class AdaptiveCruiseControl:
    """
    Define the symbolic dynamic:    dx = f(x) + g(x) * u
    x contains 3 states:  p -> position     v -> velocity    relative z -> distance


    """
    def __init__(self, params):
        """
        The input 'params' is a dictionary type argument which contains the following parameters:

        :param f0   :   To define the rolling resistance;
        :param f1   :   To define the rolling resistance;
        :param f2   :   To define the rolling resistance;
        :param m    :   The mass;
        :param v0   :   The speed of leading cruise;
        :param T    :   The time horizon for defining cbf;
        :param cd   :   The deceleration parameter in cbf;
        :param vd   :   The desired velocity in clf;
        :param udim :   The dimension of control profile u
        """
        self.f0 = params['f0']
        self.f1 = params['f1']
        self.f2 = params['f2']
        self.v0 = params['v0']
        self.m  = params['m']

        self.T  = params['T']
        self.cd = params['cd']
        self.G  = params['G']

        self.vd = params['vd']

        p, v, z = sp.symbols('p v z')
        self.x  = sp.Matrix([p, v, z])

        self.Fr = None

        # Define the symbolic expression for system dynamic, CLF and CBF
        self.f, self.g = self.simple_car_dynamics()
        self.cbf = self.define_cbf()
        self.clf = self.define_clf()

        if 'udim' in params.keys():
            self.udim = params['udim']
        else:
            print(f'The dimension of input u is not given, set it to be default 1')
            self.udim = 1

    def simple_car_dynamics(self):
        self.Fr = self.Fr_()
        # f, g both column vector
        f = sp.Matrix([self.x[1], -self.Fr / self.m, self.v0 - self.x[1]])
        g = sp.Matrix([0, 1/self.m, 0])
        return f, g

    def Fr_(self):
        self.Fr = self.f0 + self.f1 * self.x[1] + self.f2 * self.x[1] ** 2
        return self.Fr

    def getFr(self, x):
        return np.array([self.f0 + self.f1 * x[1] + self.f2 * x[1] ** 2])

    def define_cbf(self):
        cbf = self.x[2] - self.T * self.x[1] - .5 * (self.x[1] - self.v0) ** 2 / (self.cd * self.G)
        return cbf

    def define_clf(self):
        clf = (self.x[1] - self.vd) ** 2
        return clf
