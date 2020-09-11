from pysmt.typing import REAL

class SMT_Instance():
    def __init__(self):
        self.variables = {} #Variables is a dictionary e.g. namevar -> [a, b]
        self.operations = {} #Operations is a dictionary e.g. nameoperation -> [a, b]

    def encode(self):
        pass

    def add_var(self, var_name, a, b):
        self.variables[var_name]=[a,b]

    def add_expression(self, expression, a, b):
        self.operations[expression]=[a,b]

    def merge_instance(self, smt_instance):
        if isinstance(smt_instance, SMT_Instance):
            self.variables.update(smt_instance)
        else:
            print("ERROR: merging with an object of unknown type")
            return
        return self

def create_name_for_UnaryOperation_SMT_LIB(operand, operation=None):
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

def create_name_for_BinaryOperation_SMT_LIB(operand_left, operation, operand_right):
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