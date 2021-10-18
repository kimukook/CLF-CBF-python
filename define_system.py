'''
=====================================
Author  :  Muhan Zhao
Date    :  Feb. 11, 2020
Location:  UC San Diego, La Jolla, CA
=====================================
'''

import sympy as sp
from sympy.utilities.lambdify import lambdify
import numpy as np


class ControlAffineSystem:
    '''
    This class defines the dynamic of the control affine system:

            dx = f(x) + g(x) * u
    f -> 2D column-wise vector
    g -> 2D column-wise vector

    This class also includes the control barrier function (cbf) and control Lyapunov function (clf) in a symbolic way.

    This class has the following methods:

    - Compute the Lie derivative of CLF and CBF w.r.t. f(x) and g(x):



    '''
    def __init__(self, system):
        self.x = system.x  # column-wise vector
        self.xdim = system.x.shape[0]
        self.params = None
        self.xdim = self.x.shape[0]
        self.udim = system.udim

        self.f = None
        self.f_symbolic = None

        self.g = None
        self.g_symbolic = None

        self.cbf = None
        self.cbf_symbolic = None
        self.clf = None
        self.clf_symbolic = None

        # Lie derivative of clf w.r.t f as a function
        self.lf_clf = None
        self.lf_clf_symbolic = None

        # Lie derivative of clf w.r.t g as a function
        self.lg_clf = None
        self.lg_clf_symbolic = None

        # Lie derivative of cbf w.r.t f as a function
        self.lf_cbf = None
        self.lf_cbf_symbolic = None

        # Lie derivative of cbf w.r.t f as a function
        self.lg_cbf = None
        self.lg_cbf_symbolic = None

        self.define_system(system)
        self.define_cbf(system.cbf)
        self.define_clf(system.clf)
        self.lie_derivatives_calculator()

    def define_system(self, dynamic_system_class):
        # todo check f and g both are symbolic expression
        self.f_symbolic = dynamic_system_class.f
        self.f = lambdify(np.array(self.x.T), self.f_symbolic, 'numpy')
        if self.f(np.ones(self.xdim)).shape != (self.xdim, 1):
            raise ValueError(f'The output of f(x) should be (xdim, 1), now it is {self.f(np.ones(self.xdim)).shape}')

        self.g_symbolic = dynamic_system_class.g
        self.g = lambdify(np.array(self.x.T), self.g_symbolic, 'numpy')
        if self.g(np.ones(self.xdim)).shape != (self.xdim, 1):
            raise ValueError(f'The output of g(x) should be (xdim, 1), now it is {self.g(np.ones(self.xdim)).shape}')

    def define_cbf(self, cbf):
        """
        Define the symbolic control barrier function
        :param cbf:
        :return:
        """
        # todo check cbf is symbolic input
        self.cbf_symbolic = cbf
        # input for cbf has to be a 2D column-wise vector with size (self.xdim, 1)
        self.cbf = lambdify(np.array(self.x.T), self.cbf_symbolic, 'numpy')

    def define_clf(self, clf):
        """
        Define the symbolic control barrier function
        :param clf:
        :return:
        """
        # todo check clf is symbolic input
        self.clf_symbolic = clf
        # input for clf has to be a column-wise vector with size (self.xdim, 1)
        self.clf = lambdify(np.array(self.x.T), self.clf_symbolic, 'numpy')

    def lie_derivatives_calculator(self):
        """
        Compute the Lie derivatives of CBF and CLF w.r.t to x
        :return:
        """
        dx_cbf_symbolic = sp.Matrix([self.cbf_symbolic]).jacobian(self.x)

        self.lf_cbf_symbolic = dx_cbf_symbolic * self.f_symbolic
        self.lg_cbf_symbolic = dx_cbf_symbolic * self.g_symbolic

        # input for lf_cbf and lg_cbf has to be a column-wise vector with size (self.xdim, 1)
        self.lf_cbf = lambdify(np.array(self.x.T), self.lf_cbf_symbolic, 'numpy')
        self.lg_cbf = lambdify(np.array(self.x.T), self.lg_cbf_symbolic, 'numpy')

        dx_clf_symbolic = sp.Matrix([self.clf_symbolic]).jacobian(self.x)

        self.lf_clf_symbolic = dx_clf_symbolic * self.f_symbolic
        self.lg_clf_symbolic = dx_clf_symbolic * self.g_symbolic

        # input for lf_clf and lg_clf has to be a column-wise vector with size (self.xdim, 1)
        self.lf_clf = lambdify(np.array(self.x.T), self.lf_clf_symbolic, 'numpy')
        self.lg_clf = lambdify(np.array(self.x.T), self.lg_clf_symbolic, 'numpy')

    def __str__(self):
        # TODO
        return f'Class contains the states {self.x}'

    def __repr__(self):
        # TODO
        return f'Class contains the states {self.x}'

