import copy
import subprocess
from decimal import Decimal
import shlex
import sys
sys.setrecursionlimit(10000)

import numpy as np
from scipy.optimize import linprog

from SMT_Interface import clean_var_name_SMT
from mixedarithmetic import dec2Str
from project_utils import round_near
from setup_utils import eps_for_LP, digits_for_cdf


def add_minus_to_number_str(numstr):
    tmp=numstr.strip()
    if tmp[0]=="-":
        return tmp[1:]
    elif tmp[0]=="+":
        return "-"+tmp[1:]
    else:
        return "-"+tmp

'''
LP with Z3
'''
class LP_with_SMT():
    def __init__(self, left_name, right_name, marginal_left, marginal_right, insides, evaluation_points, debug=True):
        self.debug=debug
        self.marginal_left=marginal_left
        self.marginal_right=marginal_right
        self.left_name=clean_var_name_SMT(left_name)
        self.right_name=clean_var_name_SMT(right_name)
        if self.left_name==self.right_name:
            print("LP with SMT, names of operands are equal!. Using TMP_name to distinguish.")
            self.right_name="TMP_"+self.right_name
        self.evaluation_points=evaluation_points
        self.insiders=insides
        self.query = self.encode_variables()+self.encode_marginals_left()+\
              self.encode_marginals_right()+self.encode_insiders()

    '''
    Get the number from the output of Z3
    '''
    def clean_result_of_optimization(self, out):
        res = out.decode().strip()
        new_line_clean=res.replace("\n","")
        par_res = new_line_clean.split("))")[0]
        space_res=par_res.split()[-1]
        marks_clean=space_res.replace("?","")
        return str(float(marks_clean))

    def optimize_max(self):
        edge_cdf=[]
        val_cdf = []
        print("LP problem Maximize, num evaluation points= " + str(len(self.evaluation_points)))
        for index_lp, ev_point in enumerate(self.evaluation_points):
            print("Problem: "+str(index_lp))
            res_values = [intern for intern in self.insiders if (Decimal(intern.interval.lower) <= ev_point)]
            encode="\n\n(maximize "+self.encode_recursive_addition(res_values)+")"
            query=self.query+encode+"\n(check-sat)\n(get-objectives)\n"
            solver_query = "z3 pp.decimal=true -in"
            proc_run = subprocess.Popen(shlex.split(solver_query),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc_run.communicate(input=str.encode(query))
            if not err.decode() == "":
                print("Problem in the solver!")
            res=self.clean_result_of_optimization(out)
            edge_cdf.append(dec2Str(ev_point))
            val_cdf.append(round_near(Decimal(res), digits_for_cdf))
        return edge_cdf, val_cdf

    def optimize_min(self):
        edge_cdf=[]
        val_cdf = []
        print("LP problem Minimize, num evaluation points= " + str(len(self.evaluation_points)))
        for index_lp, ev_point in enumerate(self.evaluation_points):
            print("Problem: "+str(index_lp))
            res_values = [intern for intern in self.insiders if (Decimal(intern.interval.lower) <= ev_point)]
            encode="\n\n(minimize "+self.encode_recursive_addition(res_values)+")"
            query=self.query+encode+"\n(check-sat)\n(get-objectives)\n"
            solver_query = "z3 pp.decimal=true -in"
            proc_run = subprocess.Popen(shlex.split(solver_query),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc_run.communicate(input=str.encode(query))
            if not err.decode() == "":
                print("Problem in the solver!")
            res=self.clean_result_of_optimization(out)
            edge_cdf.append(dec2Str(ev_point))
            val_cdf.append(round_near(Decimal(res), digits_for_cdf))
        return edge_cdf, val_cdf

    def encode_variables(self):
        counter=0
        declare_vars=""
        for marginal_left in self.marginal_left:
            marginal_left.name=self.left_name+"_"+str(counter)
            declare_vars=declare_vars+"(declare-const "+marginal_left.name+" Real)\n"
            declare_vars=declare_vars+"(assert (<= "+marginal_left.name+" 1.0))\n"
            declare_vars = declare_vars + "(assert (>= " + marginal_left.name + " 0.0))\n"
            counter=counter+1
        declare_vars = declare_vars + "\n\n"
        counter = 0
        for marginal_right in self.marginal_right:
            marginal_right.name=self.right_name+"_"+str(counter)
            declare_vars=declare_vars+"(declare-const "+marginal_right.name+" Real)\n"
            declare_vars=declare_vars+"(assert (<= "+marginal_right.name+" 1.0))\n"
            declare_vars = declare_vars + "(assert (>= " + marginal_right.name + " 0.0))\n"
            counter=counter+1
        declare_vars = declare_vars + "\n\n"
        counter=0
        for inside in self.insiders:
            inside.name="insider_"+str(counter)
            declare_vars = declare_vars + "(declare-const " + inside.name + " Real)\n"
            declare_vars = declare_vars+"(assert (<= "+inside.name+" 1.0))\n"
            declare_vars = declare_vars + "(assert (>= " + inside.name + " 0.0))\n"
            counter = counter + 1
        declare_vars = declare_vars + "\n\n"
        return declare_vars

    def encode_recursive_addition(self, list):
        if len(list)==1:
            return list[0].name
        else:
            return "(+ "+list[0].name+" "+self.encode_recursive_addition(list[1:])+" "+")"

    '''
    Insiders have to sum up to the marginal
    '''
    def encode_insiders(self):
        encode_insiders=""
        for marginal in self.marginal_right:
            encode_insiders=encode_insiders+\
                            "(assert (= "+self.encode_recursive_addition(list(marginal.kids))+" "+marginal.name+"))\n"
        for marginal in self.marginal_left:
            encode_insiders=encode_insiders+\
                            "(assert (= "+self.encode_recursive_addition(list(marginal.kids))+" "+marginal.name+"))\n"
        return encode_insiders

    '''
    The marginals represent CDF values. So the CDF at the i marginal has to be equal 
    to the sum of all marginals from 0 to i itself.
    '''
    def encode_marginals_left(self):
        encode_marginals_left=""
        so_far_left=self.marginal_left[0].name
        encode_marginals_left = encode_marginals_left + \
                                "(assert ( <= " + self.marginal_left[0].cdf_low + " " + so_far_left + "))\n"
        encode_marginals_left = encode_marginals_left + \
                                "(assert ( >= " + self.marginal_left[0].cdf_up + " " + so_far_left + "))\n"
        for marginal in self.marginal_left[1:]:
            so_far_left="(+ "+marginal.name+" "+so_far_left+")"
            encode_marginals_left=encode_marginals_left+\
                                  "(assert ( <= "+marginal.cdf_low+" "+so_far_left+"))\n"
            encode_marginals_left=encode_marginals_left+\
                                  "(assert ( >= "+marginal.cdf_up+" "+so_far_left+"))\n"
        encode_marginals_left = encode_marginals_left + "\n\n"
        return encode_marginals_left

    def encode_marginals_right(self):
        encode_marginals_right=""
        so_far_right=self.marginal_right[0].name
        encode_marginals_right = encode_marginals_right + \
                                 "(assert ( <= " + self.marginal_right[0].cdf_low + " " + so_far_right + "))\n"
        encode_marginals_right = encode_marginals_right + \
                                 "(assert ( >= " + self.marginal_right[0].cdf_up + " " + so_far_right + "))\n"
        for marginal in self.marginal_right[1:]:
            so_far_right="(+ "+marginal.name+" "+so_far_right+")"
            encode_marginals_right=encode_marginals_right+\
                                  "(assert ( <= "+marginal.cdf_low+" "+so_far_right+"))\n"
            encode_marginals_right=encode_marginals_right+\
                                  "(assert ( >= "+marginal.cdf_up+" "+so_far_right+"))\n"
        encode_marginals_right = encode_marginals_right + "\n\n"
        return encode_marginals_right

'''
LP with Numpy.
'''
class LP_Instance:
    def __init__(self, marginal_left, marginal_right, insides, evaluation_points, debug=True):
        self.i=0
        self.debug=debug
        self.association_internals = {}
        self.association_marginals = {}
        self.constraints=[]
        self.values=[]
        self.marginal_left=marginal_left
        self.marginal_right=marginal_right
        self.internals=insides
        self.evaluation_points=set()
        self.elaborate_associations()
        self.elaborate_constraints()
        self.evaluation_points=evaluation_points

    def elaborate_associations(self):
        #Please note that marginal_left and marginal_right are deep clones (no cross references)
        for pbox in self.marginal_left:
            for kid in pbox.kids:
                if kid not in self.association_internals:
                    self.association_internals[kid]=self.i
                    self.i=self.i+1

        for pbox in self.marginal_left:
            if pbox not in self.association_marginals:
                self.association_marginals[pbox]=self.i
                self.i = self.i + 1

        for pbox in self.marginal_right:
            for kid in pbox.kids:
                if kid not in self.association_internals:
                    self.association_internals[kid] = self.i
                    self.i = self.i + 1

        for pbox in self.marginal_right:
            if pbox not in self.association_marginals:
                self.association_marginals[pbox]=self.i
                self.i = self.i + 1

    def elaborate_constraints(self):
        #cells inside the square has to sum up to marginals
        for pbox in self.marginal_left:
            vect=np.zeros(len(self.association_internals) + len(self.association_marginals))
            for kid in pbox.kids:
                vect[self.association_internals[kid]]=1
            # move to the other side of the equality
            vect[self.association_marginals[pbox]] = -1
            self.constraints.append(vect)
            self.values.append((str(eps_for_LP), str(eps_for_LP)))
        if self.debug:
            print("Constraints Insiders to Marginal Left: "+str(len(self.constraints)))
        #cells inside the square has to sum up to marginals
        for pbox in self.marginal_right:
            vect=np.zeros(len(self.association_internals) + len(self.association_marginals))
            for kid in pbox.kids:
                vect[self.association_internals[kid]]=1
            vect[self.association_marginals[pbox]] = -1
            self.constraints.append(vect)
            self.values.append((str(eps_for_LP), str(eps_for_LP)))
        if self.debug:
            print("Constraints Insiders to Marginal Right: "+str(len(self.marginal_right)))
        #cdf is in the form [0.0,0.2] so we encode >= 0.0 and <= 0.2 this is why we add minus
        vect=np.zeros(len(self.association_internals) + len(self.association_marginals))
        for i in range(0, len(self.marginal_left)):
            pbox=self.marginal_left[i]
            vect[self.association_marginals[pbox]]=1
            self.constraints.append(copy.deepcopy(vect))
            self.values.append((add_minus_to_number_str(pbox.cdf_low), pbox.cdf_up))
        if self.debug:
            print("Constraints Sums of Marginal Left: "+str(len(self.marginal_left)))
        #cdf is in the form [0.0,0.2] so we encode >= 0.0 and <= 0.2 this is why we add minus
        vect = np.zeros(len(self.association_internals) + len(self.association_marginals))
        for i in range(0, len(self.marginal_right)):
            pbox=self.marginal_right[i]
            vect[self.association_marginals[pbox]]=1
            self.constraints.append(copy.deepcopy(vect))
            self.values.append((add_minus_to_number_str(pbox.cdf_low), pbox.cdf_up))
        if self.debug:
            print("Constraints Sums of Marginal Right: "+str(len(self.marginal_right)))

    def prepare_constraints_with_inequalities(self):
        tmp_constraints = []
        tmp_b = []
        for ind, val in enumerate(self.constraints):
            tmp_constraints.append(-val)
            tmp_b.append(self.values[ind][0])
            tmp_constraints.append(val)
            tmp_b.append(self.values[ind][1])
        return tmp_constraints, tmp_b

    def optimize_max(self):
        edge_cdf=[]
        val_cdf = []
        tmp_constraints,tmp_b=self.prepare_constraints_with_inequalities()
        print("LP problem, num evaluation points= "+str(len(self.evaluation_points)))
        for index_lp, ev_point in enumerate(self.evaluation_points):
            vect=np.zeros(len(self.association_internals) + len(self.association_marginals))
            res_values = [intern for
                            intern in self.internals if (Decimal(intern.interval.lower) <= ev_point)]
            for element in res_values:
                vect[self.association_internals[element]] = 1
            res = linprog(-vect, A_ub=np.array(tmp_constraints), b_ub=np.array(tmp_b), bounds=(0, 1.0))#, method='revised simplex')
            edge_cdf.append(dec2Str(ev_point))
            res_val=min(Decimal("1.0"), max(Decimal("0.0"), Decimal(-res.fun)))
            val_cdf.append(round_near(res_val, digits_for_cdf))
        return edge_cdf, val_cdf

    def optimize_min(self):
        edge_cdf = []
        val_cdf = []
        tmp_constraints,tmp_b=self.prepare_constraints_with_inequalities()
        print("LP problem, num evaluation points= "+str(len(self.evaluation_points)))
        for index_lp, ev_point in enumerate(self.evaluation_points):
            vect=np.zeros(len(self.association_internals) + len(self.association_marginals))
            res_values = [intern for
                          intern in self.internals if (Decimal(intern.interval.upper) <= ev_point)]
            for element in res_values:
                vect[self.association_internals[element]] = 1
            res = linprog(vect, A_ub=np.array(tmp_constraints), b_ub=np.array(tmp_b), bounds=(0, 1.0))#, method='revised simplex')
            edge_cdf.append(dec2Str(ev_point))
            res_val=min(Decimal("1.0"), max(Decimal("0.0"), Decimal(res.fun)))
            val_cdf.append(round_near(res_val, digits_for_cdf))
        return edge_cdf, val_cdf