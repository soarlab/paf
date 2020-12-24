import decimal
import os
import shlex
import signal
import subprocess
from decimal import Decimal

import numpy

from project_utils import isNumeric, dec2Str
from setup_utils import hard_timeout, SMT_exponent_function_name, abs_prefix


#Interface for the communication with the SMT solver
class SMT_Instance():
    def __init__(self):
        self.variables = {} #Variables is a dictionary e.g. namevar -> [a, b]
        self.operation_left = () #Operations is a tuple e.g. nameoperation -> (expression, pbox)
        self.operation_right = () #Operations is a tuple e.g. nameoperation -> (expression, pbox)
        self.operation_center = ()  #This is used to prune de domain of Z=XopY

    def encode(self):
        res=""
        for variable in self.variables:
            res=res+self.encode_variable(variable)
        res=res+self.encode_operations()
        res=res+"(check-sat)\n" #-using (or-else default qfnra-nlsat))\n"
        return res

    '''
    Encode a variable with its range
    '''
    def encode_variable(self, variable):
        info_var=self.variables[variable] #[a, b] a and b are always included
        res=""
        res = res + "(declare-const " + variable + " Real)\n"
        res = res + "(assert (<= " + self.clean_string(info_var[0]) + " " + variable + "))\n"
        res = res + "(assert (>= " + self.clean_string(info_var[1]) + " " + variable + "))\n"
        res = res + "\n"
        return res

    '''
    The solver does not support scientific notation.
    '''
    def clean_string(self, a):
        if "e" in str(a) or "E" in str(a):
            return "{:.500f}".format(a).rstrip('0')+"0"
        return str(a)

    '''
    Encode an operation together with the pbox.
    '''
    def encode_operation(self, operation):
        #operation is a tuple (expression, p_box)
        expression = operation[0]
        p_box = operation[1]
        res = ""
        operator_left = "<=" if p_box.include_lower else "<"
        operator_right = ">=" if p_box.include_upper else ">"
        res= res +"(assert (" + operator_left + " " + self.clean_string(p_box.lower) + " " + expression + "))\n"
        res= res +"(assert (" + operator_right + " " + self.clean_string(p_box.upper) + " " + expression + "))\n"
        res = res + "\n"
        return res

    def encode_operations(self):
        tmp=""
        if not self.operation_left == ():
            tmp=tmp+self.encode_operation(self.operation_left)
        if not self.operation_right == ():
            tmp=tmp+self.encode_operation(self.operation_right)
        if not self.operation_center == ():
            tmp=tmp+self.encode_operation(self.operation_center)
        return tmp

    '''
    dreal is delta sat with a certain imprecision: usually is better for range analysis.
    z2 is better for error analysis, where the domain of the operation is in the order of e-10.
    '''
    def check(self, debug=False, dReal=True):
        prelude=""
        #Logic is non-linear real arithmetic
        prelude=prelude+"(set-logic QF_NRA)\n"
        query=prelude+self.encode()
        if debug:
            print(query)
        if dReal:
            solver_query = "dreal --in"
        else:
            solver_query = "z3 -in"

        proc_run = subprocess.Popen(shlex.split(solver_query), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = proc_run.communicate(input=str.encode(query), timeout=hard_timeout)
            if not err.decode() == "":
                print("Problem in the solver!")
            res = out.decode().strip()
        except subprocess.TimeoutExpired:
            try:
                os.kill(proc_run.pid, signal.SIGINT) # send signal to the process group
                os.killpg(proc_run.pid, signal.SIGINT) # send signal to the process group
            except OSError:
                # silently fail if the subprocess has exited already
                pass
            res="timeout"
        if debug:
            print(res)
        if res=="sat" or "delta-sat with delta" in res:
            return 1
        elif res=="unknown" or res=="timeout":
            print("timeout/unknown from the solver. dReal ?"+str(dReal))
            if debug:
                print(query)
            return 1
        elif res=="unsat":
            return 0
        else:
            print("Misterious output from the solver!!!\n\n\n")
            exit(-1)

    def add_var(self, var_name, a, b):
        if not isNumeric(var_name):
            self.variables[var_name]=[a,b]

    def set_expression_left(self, expression, p_box):
        self.operation_left=(expression, p_box)

    def clean_expressions(self):
        self.operation_left=()
        self.operation_right=()
        self.operation_center=()

    def set_expression_right(self, expression, p_box):
        self.operation_right=(expression, p_box)

    def set_expression_central(self, expression, p_box):
        self.operation_center=(expression, p_box)

    def merge_instance(self, smt_instance):
        if isinstance(smt_instance, SMT_Instance):
            self.variables.update(smt_instance.variables)
        else:
            print("ERROR: merging with an object of unknown type")
            return
        return self

class Precise_SMT_Instance():

    def __init__(self):
        self.variables = {} #Variables is a dictionary e.g. namevar -> [a, b]
        self.abs_prelude = {}
        self.operation_left = () #Operations is a tuple e.g. nameoperation -> (expression, pbox)
        self.operation_right = () #Operations is a tuple e.g. nameoperation -> (expression, pbox)
        self.operation_center = ()  #This is used to prune de domain of Z=XopY

    def binarySearch(self, arr, l, r):
        if l < r:
            mid = l + (r - l) / 2
            return ("(ite (< x " +
                    dec2Str(decimal.getcontext().power(2, arr[mid])) + ")\n" +
                    self.binarySearch(arr, l, mid) + "\n" +
                    self.binarySearch(arr, mid + 1, r) + ")")

        if r == l:
            return dec2Str(decimal.getcontext().power(2, arr[r - 1]))

    def encodeBinarySearch(self):
        res="(define-fun " + SMT_exponent_function_name + " ((x Real)) Real\n"
        #This works only for single floating point!!!!
        vector=list(range(-126,+127))
        res=res+self.binarySearch(vector, 0, len(vector)-1)
        res=res+")"
        return res


    def encode(self):
        res=""
        res=res+self.encodeBinarySearch()
        for variable in self.variables:
            res=res+self.encode_variable(variable)
        res=res+self.encode_operations()
        res=res+"(check-sat)\n" #-using (or-else default qfnra-nlsat))\n"
        return res

    def encode_prelude(self, var):
        res="(assert (=> (<= 0 " + var + ") (= " + self.abs_prelude[var] + " ( - 0.0 " + var + "))))\n"
        res=res+"(assert (=> (> 0 " + var + ") (= " + self.abs_prelude[var] + " " + var + ")))\n"
        return res

    '''
    Encode a variable with its range
    '''
    def encode_variable(self, variable):
        info_var=self.variables[variable] #[a, b] a and b are always included
        res=""
        res = res + "(declare-const " + variable + " Real)\n"
        res = res + "(assert (<= " + self.clean_string(info_var[0]) + " " + variable + "))\n"
        res = res + "(assert (>= " + self.clean_string(info_var[1]) + " " + variable + "))\n"
        if variable in self.abs_prelude:
            res = res + self.encode_prelude(variable)
        res = res + "\n"
        return res

    '''
    The solver does not support scientific notation.
    '''
    def clean_string(self, a):
        if "e" in str(a) or "E" in str(a):
            return "{:.500f}".format(a).rstrip('0')+"0"
        return str(a)

    '''
    Encode an operation together with the pbox.
    '''
    def encode_operation(self, operation):
        #operation is a tuple (expression, p_box)
        expression = operation[0]
        p_box = operation[1]
        res = ""
        operator_left = "<=" if p_box.include_lower else "<"
        operator_right = ">=" if p_box.include_upper else ">"
        res= res +"(assert (" + operator_left + " " + self.clean_string(p_box.lower) + " " + expression + "))\n"
        res= res +"(assert (" + operator_right + " " + self.clean_string(p_box.upper) + " " + expression + "))\n"
        res = res + "\n"
        return res

    def encode_operations(self):
        tmp=""
        if not self.operation_left == ():
            tmp=tmp+self.encode_operation(self.operation_left)
        if not self.operation_right == ():
            tmp=tmp+self.encode_operation(self.operation_right)
        if not self.operation_center == ():
            tmp=tmp+self.encode_operation(self.operation_center)
        return tmp

    '''
    dreal is delta sat with a certain imprecision: usually is better for range analysis.
    z2 is better for error analysis, where the domain of the operation is in the order of e-10.
    '''
    def check(self, debug=False, dReal=True):
        prelude=""
        #Logic is non-linear real arithmetic
        prelude=prelude+"(set-logic QF_NRA)\n"
        query=prelude+self.encode()
        if debug:
            print(query)
        if dReal:
            solver_query = "dreal --in"
        else:
            solver_query = "z3 -in"

        proc_run = subprocess.Popen(shlex.split(solver_query), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = proc_run.communicate(input=str.encode(query), timeout=hard_timeout)
            if not err.decode() == "":
                print("Problem in the solver!")
            res = out.decode().strip()
        except subprocess.TimeoutExpired:
            try:
                os.kill(proc_run.pid, signal.SIGINT) # send signal to the process group
                os.killpg(proc_run.pid, signal.SIGINT) # send signal to the process group
            except OSError:
                # silently fail if the subprocess has exited already
                pass
            res="timeout"
        if debug:
            print(res)
        if res=="sat" or "delta-sat with delta" in res:
            return 1
        elif res=="unknown" or res=="timeout":
            print("timeout/unknown from the solver")
            if debug:
                print(query)
            return 1
        elif res=="unsat":
            return 0
        else:
            print("Misterious output from the solver!!!\n\n\n")
            exit(-1)

    def add_var(self, var_name, a, b, add_also_abs=False):
        if not isNumeric(var_name):
            self.variables[var_name]=[a,b]
            if add_also_abs:
                if Decimal(a)<Decimal("0.0")<Decimal(b):
                    tmp_a = "0.0"
                    tmp_b = dec2Str(max(Decimal(a).copy_abs(),Decimal(b).copy_abs()))
                else:
                    tmp_a = dec2Str(min(Decimal(a).copy_abs(), Decimal(b).copy_abs()))
                    tmp_b = dec2Str(max(Decimal(a).copy_abs(), Decimal(b).copy_abs()))
                self.variables[abs_prefix+var_name]=[tmp_a,tmp_b]
                self.abs_prelude[var_name]=abs_prefix+var_name

    def set_expression_left(self, expression, p_box):
        self.operation_left=(expression, p_box)

    def clean_expressions(self):
        self.operation_left=()
        self.operation_right=()
        self.operation_center=()

    def set_expression_right(self, expression, p_box):
        self.operation_right=(expression, p_box)

    def set_expression_central(self, expression, p_box):
        self.operation_center=(expression, p_box)

    def merge_instance(self, smt_instance):
        if isinstance(smt_instance, SMT_Instance):
            self.variables.update(smt_instance.variables)
        else:
            print("ERROR: merging with an object of unknown type")
            return
        return self

class PBoxSolver:
    def __init__(self, lower, upper, include_lower, include_upper):
        self.lower = lower
        self.upper = upper
        self.include_lower = include_lower
        self.include_upper = include_upper

def create_exp_for_UnaryOperation_SMT_LIB(operand, operation=None):
    res=""
    if operation is not None:
        if operation == "exp":
            res = "(^ 2.7182818284590452353602874713527 " + operand + ")"
        elif operation == "cos":
            res = "(cos " + operand + ")"
        elif operation == "sin":
            res = "(sin " + operand + ")"
        elif operation == "abs":
            res = "(abs " + operand + ")"
        else:
            print("Unknown operation for the solver!")
            return "()"
    else:
        if isNumeric(operand):
            pass
        res=operand
    return res

def create_exp_for_BinaryOperation_SMT_LIB(operand_left, operation, operand_right):
    if operation == "+":
        res="(+ "+operand_left+" "+operand_right+")"
    elif operation == "-":
        res="(- "+operand_left+" "+operand_right+")"
    elif operation == "*":
        res="(* "+operand_left+" "+operand_right+")"
    elif operation == "/":
        res="(/ "+operand_left+" "+operand_right+")"
    elif operation == "*+":
        res="(* "+operand_left+" (+ 1.0 "+operand_right+"))"
    return res

def precise_create_exp_for_BinaryOperation_SMT_LIB(operand_left, operation, operand_right):
    if operation == "+":
        res="(+ "+operand_left+" "+operand_right+")"
    elif operation == "-":
        res="(- "+operand_left+" "+operand_right+")"
    elif operation == "*":
        res="(* "+operand_left+" "+operand_right+")"
    elif operation == "/":
        res="(/ "+operand_left+" "+operand_right+")"
    elif operation == "*+":
        res="(+ " + operand_left +" (* " + SMT_exponent_function_name + " " + abs_prefix + operand_left + " " + operand_right + "))"
    return res

def clean_var_name_SMT(name):
    clean_name=name.replace("(","$").replace(")","$").replace(".","dot")
    return clean_name

def make_expression_for_dict(dict):
    if len(dict) == 1:
        key = list(dict.keys())[0]
        val = dict[key]
        return "(* "+key+" "+val+")"
    else:
        tmp=list(dict.items())
        res="(* "+tmp[0][0]+" "+tmp[0][1]+")"
        for entry in tmp[1:]:
            res = "(+ (* " + entry[0] + " " + entry[1] + ") " + res + ")"
        return res

def create_expression_for_multiplication(name_dictionary_left, name_dictionary_right):
    if len(name_dictionary_left)==0 or len(name_dictionary_right)==0:
        return "0.0"
    left=make_expression_for_dict(name_dictionary_left)
    right=make_expression_for_dict(name_dictionary_right)
    return "(* "+left+" "+right+")"
