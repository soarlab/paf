import subprocess
from decimal import Decimal
import shlex
import sys
from multiprocessing.pool import Pool

sys.setrecursionlimit(100000)

from SMT_Interface import clean_var_name_SMT
from mixedarithmetic import dec2Str
from project_utils import round_near, round_up, round_down
from setup_utils import eps_for_LP, digits_for_Z3_cdf, num_processes_dependent_operation


def add_minus_to_number_str(numstr):
    tmp=numstr.strip()
    if tmp[0]=="-":
        return tmp[1:]
    elif tmp[0]=="+":
        return "-"+tmp[1:]
    else:
        return "-"+tmp

def min_instance(index_lp, ev_point, insiders, query):
    print("Problem: " + str(index_lp))
    res_values = [intern for intern in insiders if (Decimal(intern.interval.upper) <= ev_point)]
    if len(res_values) > 0:
        encode = "\n\n(minimize " + LP_with_SMT.encode_recursive_addition(res_values) + ")"
        query = query + encode + "\n(check-sat)\n(get-objectives)\n"
        solver_query = "z3 pp.decimal=true -in"
        proc_run = subprocess.Popen(shlex.split(solver_query),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc_run.communicate(input=str.encode(query))
        if not err.decode() == "":
            print("Problem in the solver!")
        res = out
    else:
        res = "0.0"
    return [ev_point, res]

def max_instance(index_lp, ev_point, insiders, query):
    print("Problem: " + str(index_lp))
    res_values = [intern for intern in insiders if (Decimal(intern.interval.lower) <= ev_point)]
    encode = "\n\n(maximize " + LP_with_SMT.encode_recursive_addition(res_values) + ")"
    query = query + encode + "\n(check-sat)\n(get-objectives)\n"
    solver_query = "z3 pp.decimal=true -in"
    proc_run = subprocess.Popen(shlex.split(solver_query),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc_run.communicate(input=str.encode(query))
    if not err.decode() == "":
        print("Problem in the solver!")
        exit(-1)
    return [ev_point, out]

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
    @staticmethod
    def clean_result_of_optimization(out):
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

        pool = Pool(processes=num_processes_dependent_operation//2, maxtasksperchild=3)
        tmp_results=[]
        results=[]
        for index_lp, ev_point in enumerate(self.evaluation_points):
            tmp_results.append(pool.apply_async(max_instance,
                                        args=[index_lp,ev_point,self.insiders,self.query],
                                        callback=results.append))
        pool.close()
        pool.join()

        for pair in sorted(results, key=lambda x: x[0]):
            res=self.clean_result_of_optimization(pair[1])
            edge_cdf.append(dec2Str(pair[0]))
            val_cdf.append(round_near(Decimal(res), digits_for_Z3_cdf))

        return edge_cdf, val_cdf

    def optimize_min(self):
        edge_cdf=[]
        val_cdf = []
        print("LP problem Minimize, num evaluation points= " + str(len(self.evaluation_points)))

        pool = Pool(processes=num_processes_dependent_operation//2, maxtasksperchild=3)
        tmp_results = []
        results = []

        for index_lp, ev_point in enumerate(self.evaluation_points):
            tmp_results.append(pool.apply_async(min_instance,
                                        args=[index_lp,ev_point,self.insiders,self.query],
                                        callback=results.append))

        pool.close()
        pool.join()

        for pair in sorted(results, key=lambda x: x[0]):
            if pair[1]=="0.0":
                res="0.0"
            else:
                res = self.clean_result_of_optimization(pair[1])
            edge_cdf.append(dec2Str(pair[0]))
            val_cdf.append(round_near(Decimal(res), digits_for_Z3_cdf))

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

    @staticmethod
    def encode_recursive_addition(tmp_list):
        if len(tmp_list)==1:
            return tmp_list[0].name
        else:
            return "(+ " + tmp_list[0].name + " " + LP_with_SMT.encode_recursive_addition(tmp_list[1:]) + " " + ")"

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
                                "(assert ( <= " + dec2Str(round_down(self.marginal_left[0].cdf_low, digits_for_Z3_cdf)) + " " + so_far_left + "))\n"
        encode_marginals_left = encode_marginals_left + \
                                "(assert ( >= " + dec2Str(round_up(self.marginal_left[0].cdf_up, digits_for_Z3_cdf)) + " " + so_far_left + "))\n"
        for marginal in self.marginal_left[1:]:
            so_far_left="(+ "+marginal.name+" "+so_far_left+")"
            encode_marginals_left=encode_marginals_left+\
                                  "(assert ( <= "+ dec2Str(round_down(marginal.cdf_low, digits_for_Z3_cdf))+" "+so_far_left+"))\n"
            encode_marginals_left=encode_marginals_left+\
                                  "(assert ( >= "+ dec2Str(round_up(marginal.cdf_up, digits_for_Z3_cdf))+" "+so_far_left+"))\n"
        encode_marginals_left = encode_marginals_left + "\n\n"
        return encode_marginals_left

    def encode_marginals_right(self):
        encode_marginals_right=""
        so_far_right=self.marginal_right[0].name
        encode_marginals_right = encode_marginals_right + \
                                 "(assert ( <= " + dec2Str(round_down(self.marginal_right[0].cdf_low, digits_for_Z3_cdf)) + " " + so_far_right + "))\n"
        encode_marginals_right = encode_marginals_right + \
                                 "(assert ( >= " + dec2Str(round_up(self.marginal_right[0].cdf_up, digits_for_Z3_cdf)) + " " + so_far_right + "))\n"
        for marginal in self.marginal_right[1:]:
            so_far_right="(+ "+marginal.name+" "+so_far_right+")"
            encode_marginals_right=encode_marginals_right+\
                                  "(assert ( <= "+dec2Str(round_down(marginal.cdf_low, digits_for_Z3_cdf))+" "+so_far_right+"))\n"
            encode_marginals_right=encode_marginals_right+\
                                  "(assert ( >= "+dec2Str(round_up(marginal.cdf_up, digits_for_Z3_cdf))+" "+so_far_right+"))\n"
        encode_marginals_right = encode_marginals_right + "\n\n"
        return encode_marginals_right