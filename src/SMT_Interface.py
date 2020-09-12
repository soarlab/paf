import shlex
import subprocess


class SMT_Instance():
    def __init__(self):
        self.variables = {} #Variables is a dictionary e.g. namevar -> [a, b]
        self.operation_left = () #Operations is a dictionary e.g. nameoperation -> [a, b]
        self.operation_right = ()  # Operations is a dictionary e.g. nameoperation -> [a, b]

    def encode(self):
        res=""
        for variable in self.variables:
            res=res+self.encode_variable(variable,self.variables[variable][0],self.variables[variable][1])
        res=res+self.encode_expression(self.operation_left[0], self.operation_left[1], self.operation_left[2])
        res=res+self.encode_expression(self.operation_right[0], self.operation_right[1], self.operation_right[2])
        res=res+"(check-sat-using (or-else default qfnra-nlsat))\n"
        return res

    def encode_variable(self, variable, a, b):
        res=""
        res = res + "(declare-const " + variable + " Real)\n"
        res = res + "(assert (< " + self.clean_float(a) + " " + variable + "))\n"
        res = res + "(assert (> " + self.clean_float(b) + " " + variable + "))\n"
        res = res + "\n"
        return res

    def clean_float(self, a):
        if "e" in str(a) or "E" in str(a):
            return "{:.500f}".format(a).rstrip('0')+"0"
        return str(a)

    def encode_expression(self, expression, a, b):
        res=""
        res=res+"(assert (< " + str(a) + " " + expression + "))\n"
        res=res+"(assert (> " + str(b) + " " + expression + "))\n"
        res = res + "\n"
        return res

    def check(self):
        query=self.encode()
        z3_query="z3 -in"
        proc_run = subprocess.Popen(shlex.split(z3_query),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc_run.communicate(input=str.encode(query))
        if not err.decode()=="":
            print("Problem in the solver!")
        res=out.decode().strip()
        if res=="sat" or res=="unknown":
            return 1
        elif res=="unsat":
            return 0
        else:
            print("Misterious output from the solver")
            return 1

    def add_var(self, var_name, a, b):
        self.variables[var_name]=[a,b]

    def set_expression_left(self, expression, a, b):
        self.operation_left=(expression, a, b)

    def set_expression_right(self, expression, a, b):
        self.operation_right=(expression, a, b)

    def merge_instance(self, smt_instance):
        if isinstance(smt_instance, SMT_Instance):
            self.variables.update(smt_instance.variables)
        else:
            print("ERROR: merging with an object of unknown type")
            return
        return self

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

def clean_var_name_SMT(name):
    clean_name=name.replace("(","$").replace(")","$")
    return clean_name