import giskardpy.symengine_wrappers as spw
import hashlib
import numpy as np
import pickle

from collections import OrderedDict, namedtuple
from itertools import chain
from time import time

from giskardpy import BACKEND
from giskardpy import print_wrapper
from giskardpy.exceptions import QPSolverException
from giskardpy.qp_solver import QPSolver
from giskardpy.symengine_wrappers import load_compiled_function, safe_compiled_function


SoftConstraint = namedtuple('SoftConstraint', ['lower', 'upper', 'weight', 'expression'])
HardConstraint = namedtuple('HardConstraint', ['lower', 'upper', 'expression'])
JointConstraint = namedtuple('JointConstraint', ['lower', 'upper', 'weight'])

BIG_NUMBER = 1e9

def subs_if_sym(var, subs_dict):
    t = type(var)
    if t == int or t == float or t == str:
        return var
    else:
        return var.subs(subs_dict)

def pretty_matrix_format_str(col_names, row_names, min_col_width=10):
    if len(row_names) > 1:
        w_first_col = max(*[len(n) for n in row_names])
    else:
        w_first_col = len(row_names[0])
    widths = [max(min_col_width, len(c)) for c in col_names]

    out = ''.join([(' ' * w_first_col)] + ['  {:>{:d}}'.format(n, w) for n, w in zip(col_names, widths)])
    for y in range(len(row_names)):
        out += '\n{:>{:d}}'.format(row_names[y], w_first_col)
        out += ''.join([', {}:>{:d}.5{}'.format('{', w, '}') for w in widths])

    return out

def format_matrix(matrix, mat_str):
    return mat_str.format(*matrix.reshape(1, matrix.shape[0] * matrix.shape[1]).tolist()[0])


class QProblemBuilder(object):
    def __init__(self, joint_constraints_dict, hard_constraints_dict, soft_constraints_dict, controlled_joint_symbols,
                 free_symbols=None, path_to_functions='', print_fn=print_wrapper):
        assert (not len(controlled_joint_symbols) > len(joint_constraints_dict))
        assert (not len(controlled_joint_symbols) < len(joint_constraints_dict))
        assert (len(hard_constraints_dict) <= len(controlled_joint_symbols))
        self.path_to_functions = path_to_functions
        self.free_symbols = free_symbols
        self.joint_constraints_dict = joint_constraints_dict
        self.hard_constraints_dict = hard_constraints_dict
        self.soft_constraints_dict = soft_constraints_dict
        self.controlled_joints = controlled_joint_symbols
        self.controlled_joints_strs = [str(x) for x in self.controlled_joints]
        self.__print_fn = print_fn
        self.soft_constraint_indices = {}
        self.make_sympy_matrices()

        self.shape1 = len(self.hard_constraints_dict) + len(self.soft_constraints_dict)
        self.shape2 = len(self.joint_constraints_dict) + len(self.soft_constraints_dict)

        self.qp_solver = QPSolver(len(self.joint_constraints_dict) + len(self.soft_constraints_dict),
                                  len(self.hard_constraints_dict) + len(self.soft_constraints_dict))

    # @profile
    def make_sympy_matrices(self):
        t_total = time()
        # TODO cpu intensive
        weights = []
        lb = []
        ub = []
        lbA = []
        ubA = []
        soft_expressions = []
        hard_expressions = []
        for k, c in self.joint_constraints_dict.items():
            weights.append(c.weight)
            lb.append(c.lower)
            ub.append(c.upper)
        for k, c in self.hard_constraints_dict.items():
            lbA.append(c.lower)
            ubA.append(c.upper)
            hard_expressions.append(c.expression)
        for k, c in self.soft_constraints_dict.items():
            self.soft_constraint_indices[k] = len(lbA)
            weights.append(c.weight)
            lbA.append(c.lower)
            ubA.append(c.upper)
            lb.append(-BIG_NUMBER)
            ub.append(BIG_NUMBER)
            assert not isinstance(c.expression, spw.Matrix), 'Matrices are not allowed as soft constraint expression'
            soft_expressions.append(c.expression)
        a = ''.join(str(x) for x in sorted(chain(self.soft_constraints_dict.keys(),
                                                 self.hard_constraints_dict.keys(),
                                                 self.joint_constraints_dict.keys())))
        function_hash = None #hashlib.md5(a).hexdigest()
        self.cython_big_ass_M = None #load_compiled_function(self.path_to_functions + function_hash)
        self.np_g = np.zeros(len(weights))

        if self.cython_big_ass_M is None:
            self.__print_fn('new controller requested; compiling')
            self.H = spw.diag(*weights)

            self.lb = spw.Matrix(lb)
            self.ub = spw.Matrix(ub)

            # make A
            # hard part
            M_controlled_joints = spw.Matrix(self.controlled_joints)
            A_hard = spw.Matrix(hard_expressions)
            A_hard = A_hard.jacobian(M_controlled_joints)
            zerosHxS = spw.zeros(A_hard.shape[0], len(soft_expressions))
            A_hard = A_hard.row_join(zerosHxS)

            # soft part
            A_soft = spw.Matrix(soft_expressions)
            t = time()
            A_soft = A_soft.jacobian(M_controlled_joints)
            self.__print_fn('jacobian took {}'.format(time() - t))
            identity = spw.eye(A_soft.shape[0])
            A_soft = A_soft.row_join(identity)

            # final A
            self.A = A_hard.col_join(A_soft)

            self.lbA = spw.Matrix(lbA)
            self.ubA = spw.Matrix(ubA)

            big_ass_M_A = self.A.row_join(self.lbA).row_join(self.ubA)
            big_ass_M_H = self.H.row_join(self.lb).row_join(self.ub)
            # putting everything into one big matrix to take full advantage of cse in speed_up()
            self.big_ass_M = big_ass_M_A.col_join(big_ass_M_H)

            t = time()
            if self.free_symbols is None:
                self.free_symbols = self.big_ass_M.free_symbols
            self.cython_big_ass_M = spw.speed_up(self.big_ass_M, self.free_symbols, backend=BACKEND)
            if function_hash is not None:
                safe_compiled_function(self.cython_big_ass_M, self.path_to_functions + function_hash)
            self.__print_fn('autowrap took {}'.format(time() - t))
        else:
            self.__print_fn('controller loaded {}'.format(self.path_to_functions + function_hash))
        self.__print_fn('controller ready {}s'.format(time() - t_total))

        col_names = self.controlled_joints_strs + ['slack'] * len(self.soft_constraints_dict)
        row_names = self.hard_constraints_dict.keys() + self.soft_constraints_dict.keys()

        self.str_A  = pretty_matrix_format_str(col_names, row_names, min_col_width=20)
        self.str_b  = pretty_matrix_format_str(['lb', 'ub'], self.controlled_joints_strs + self.soft_constraints_dict.keys(), min_col_width=20)
        self.str_bA = pretty_matrix_format_str(['lbA', 'ubA'], self.hard_constraints_dict.keys() + self.soft_constraints_dict.keys(), min_col_width=20)
        self.str_xdot = pretty_matrix_format_str(['xdot'], col_names[:len(self.controlled_joints_strs)], min_col_width=20)


    def save_pickle(self, hash, f):
        with open('/tmp/{}'.format(hash), 'w') as file:
            pickle.dump(f, file)

    def load_pickle(self, hash):
        return pickle.load(hash)

    def debug_print(self, np_H, np_A, np_lb, np_ub, np_lbA, np_ubA, xdot_full):
        import pandas as pd
        lb = []
        ub = []
        lbA = []
        ubA = []
        weights = []
        xdot = []
        for iJ, k in enumerate(self.joint_constraints_dict.keys()):
            key = 'j -- ' + str(k)
            lb.append(key)
            ub.append(key)
            weights.append(key)
            xdot.append(key)

        for iH, k in enumerate(self.hard_constraints_dict.keys()):
            key = 'h -- ' + str(k)
            lbA.append(key)
            ubA.append(key)

        for iS, k in enumerate(self.soft_constraints_dict.keys()):
            key = 's -- ' + str(k)
            lbA.append(key)
            ubA.append(key)
            weights.append(key)
            xdot.append(key)
        p_lb = pd.DataFrame(np_lb[:-len(self.soft_constraints_dict)], lb)
        p_ub = pd.DataFrame(np_ub[:-len(self.soft_constraints_dict)], ub)
        p_lbA = pd.DataFrame(np_lbA, lbA)
        p_ubA = pd.DataFrame(np_ubA, ubA)
        p_weights = pd.DataFrame(np_H.dot(np.ones(np_H.shape[0])), weights)
        p_xdot = pd.DataFrame(xdot_full, xdot)
        p_A = pd.DataFrame(np_A, lbA, weights)
        pass

    def get_cmd(self, substitutions, nWSR=None):
        """

        :param substitutions: symbol -> value
        :type substitutions: dict
        :return: joint name -> joint command
        :rtype: dict
        """
        np_big_ass_M = self.cython_big_ass_M(**substitutions)
        np_H = np.array(np_big_ass_M[self.shape1:, :-2])
        np_A = np.array(np_big_ass_M[:self.shape1, :self.shape2])
        np_lb = np.array(np_big_ass_M[self.shape1:, -2])
        np_ub = np.array(np_big_ass_M[self.shape1:, -1])
        self.np_lbA = np.array(np_big_ass_M[:self.shape1, -2])
        self.np_ubA = np.array(np_big_ass_M[:self.shape1, -1])
        try:
            xdot_full = self.qp_solver.solve(np_H, self.np_g, np_A, np_lb, np_ub, self.np_lbA, self.np_ubA, nWSR)
        except QPSolverException as e:
            print('INFEASIBLE INITIAL CONFIGURATION!\n - Hard Constraints -\n{}\n - Soft Constraints -\n{}'.format('\n'.join(['{:>30}: {:>20} {:>20} {:>20}'.format(n, 
                                                                   subs_if_sym(c.lower, substitutions),
                                                                   subs_if_sym(c.upper, substitutions),
                                                                   subs_if_sym(c.expression, substitutions)) for n, c in self.hard_constraints_dict.items()]),
                       '\n'.join(['{:>30}: {:>20} {:>20} {:>20}'.format(n, 
                                                                   subs_if_sym(c.lower, substitutions),
                                                                   subs_if_sym(c.upper, substitutions),
                                                                   subs_if_sym(c.expression, substitutions)) for n, c in self.soft_constraints_dict.items()]),))
            raise e
        if xdot_full is None:
            return None
        # TODO enable debug print in an elegant way
        # self.debug_print(np_H, np_A, np_lb, np_ub, np_lbA, np_ubA, xdot_full)

        # self.__print_fn()
        # self.__print_fn('{}\n\n{}\n\n{}\n'.format(
        # self.__print_fn(format_matrix(np.concatenate((self.np_lbA.reshape(len(self.np_lbA), 1),
        #                                       self.np_ubA.reshape(len(self.np_lbA), 1)), axis=1), self.str_bA))#,
        #         '\n'.join(['{:>30}: {:>20} {:>20} {:>20}'.format(n, 
        #                                                            subs_if_sym(c.lower, substitutions),
        #                                                            subs_if_sym(c.upper, substitutions),
        #                                                            subs_if_sym(c.expression, substitutions)) for n, c in self.soft_constraints_dict.items()]),
        # self.__print_fn(format_matrix(np.concatenate((np_lb.reshape(len(np_lb), 1),
        #                               np_ub.reshape(len(np_ub), 1)), axis=1), self.str_b))
        #         format_matrix(np.array([[xdot_full[x]] for x in range(len(self.controlled_joints_strs))]), self.str_xdot)))
        return OrderedDict((observable, xdot_full[i]) for i, observable in enumerate(self.controlled_joints_strs))

    def constraints_met(self, lbThreshold=0.01, ubThreshold=-0.01, names=None):
        if names == None:
            for x in range(len(self.np_lbA)):
                if self.np_lbA[x] > lbThreshold or self.np_ubA[x] < ubThreshold:
                    return False
        else:
            for name in names:
                x = self.soft_constraint_indices[name]
                if self.np_lbA[x] > lbThreshold or self.np_ubA[x] < ubThreshold:
                    return False
        return True

    def get_a_bounds(self, name):
        if name in self.soft_constraint_indices:
            x = self.soft_constraint_indices[name]
            return (self.np_lbA[x], self.np_ubA[x])
        raise Exception('Soft constraint "{}" does not exist.'.format(name))