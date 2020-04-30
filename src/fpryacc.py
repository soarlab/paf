# from __builtin__ import str

import ply.yacc as yacc

from fprlex import *
from model import *

uniqueLexer = FPRlex()


class FPRyacc:
    tokens = FPRlex.tokens
    reserved = FPRlex.reserved

    def __init__(self, program, debug):
        self.lexer = uniqueLexer
        self.manager = NodeManager()
        self.parser = yacc.yacc(module=self)
        self.debug = debug
        self.program = program
        self.variables = {}
        self.leaves = []
        self.expression = self.parser.parse(self.program)

    def addVariable(self, name, distribution):
        if name in self.variables:
            print("Variable: " + name + " declared twice!")
            exit(-1)
        else:
            self.variables[name] = distribution

    def myPrint(self, s, p):
        res = ""
        if self.debug:
            print(str(s) + ": " + str(p[0]))
        return

    def p_FileInput(self, p):
        ''' FileInput : VarDeclaration NEWLINE Expression
		'''
        p[0] = p[3]
        self.myPrint("FileInput", p)

    def p_VarDeclaration(self, p):
        ''' VarDeclaration : Distribution
						   | Distribution COMMA VarDeclaration
		'''
        # if len(p)>2:
        #	p[0]=str(p[1])+str(p[2])+str(p[3])
        # elif len(p)==2:
        #	p[0]=str(p[1])
        self.myPrint("VarDeclaration", p)

    def p_Edges(self,p):
        ''' Edge : POSNUMBER
                 | POSNUMBER COMMA Edge
        '''

        if len(p) > 3:
            p[0] = [float(p[1])]+p[3]
        else:
            p[0]=[float(p[1])]

    def p_Values(self,p):
        ''' Value : POSNUMBER
                  | POSNUMBER COMMA Edge
        '''

        if len(p) > 3:
            p[0] = [float(p[1])]+p[3]
        else:
            p[0]=[float(p[1])]

    def p_Custom(self, p):
        ''' Distribution : WORD COLON C LPAREN SLPAREN Edge SRPAREN COMMA SLPAREN Value SRPAREN RPAREN
    	'''

        distr = CustomDistr(str(p[1]), p[6], p[10])
        self.addVariable(str(p[1]), distr)
        self.myPrint("Custom", p)

    def p_Uniform(self, p):
        ''' Distribution : WORD COLON U LPAREN POSNUMBER COMMA POSNUMBER RPAREN
		'''

        distr = U(str(p[1]), str(p[5]), str(p[7]))
        self.addVariable(str(p[1]), distr)
        self.myPrint("Uniform", p)

    def p_Uniform1(self, p):
        ''' Distribution : WORD COLON U LPAREN MINUS POSNUMBER COMMA POSNUMBER RPAREN
		'''

        distr = U(str(p[1]), "-" + str(p[6]), str(p[8]))
        self.addVariable(str(p[1]), distr)
        self.myPrint("Normal", p)

    def p_Uniform2(self, p):
        ''' Distribution : WORD COLON U LPAREN MINUS POSNUMBER COMMA MINUS POSNUMBER RPAREN
		'''

        distr = U(str(p[1]), "-" + str(p[6]), "-" + str(p[9]))
        self.addVariable(str(p[1]), distr)
        self.myPrint("Normal", p)

    def p_Normal(self, p):
        ''' Distribution : WORD COLON N LPAREN POSNUMBER COMMA POSNUMBER RPAREN
		'''

        distr = N(str(p[1]), str(p[5]), str(p[7]))
        self.addVariable(str(p[1]), distr)
        self.myPrint("Normal", p)

    def p_Normal1(self, p):
        ''' Distribution : WORD COLON N LPAREN MINUS POSNUMBER COMMA POSNUMBER RPAREN
		'''

        distr = N(str(p[1]), "-" + str(p[6]), str(p[8]))
        self.addVariable(str(p[1]), distr)
        self.myPrint("Normal", p)

    def p_Normal2(self, p):
        ''' Distribution : WORD COLON N LPAREN MINUS POSNUMBER COMMA MINUS POSNUMBER RPAREN
		'''

        distr = N(str(p[1]), "-" + str(p[6]), "-" + str(p[9]))
        self.addVariable(str(p[1]), distr)
        self.myPrint("Normal", p)

    def p_Beta(self, p):
        ''' Distribution : WORD COLON B LPAREN POSNUMBER COMMA POSNUMBER RPAREN
		'''

        distr = B(str(p[1]), str(p[5]), str(p[7]))
        self.addVariable(str(p[1]), distr)
        self.myPrint("Normal", p)

    def p_Expression(self, p):
        '''Expression : AnnidateArithExpr
					  | BinaryArithExpr
		'''
        p[0] = p[1]
        self.myPrint("Expression", p)

    def p_BinaryArithExpr(self, p):
        '''BinaryArithExpr : AnnidateArithExpr PLUS  AnnidateArithExpr
						   | AnnidateArithExpr MINUS AnnidateArithExpr
						   | AnnidateArithExpr MUL AnnidateArithExpr
						   | AnnidateArithExpr DIVIDE AnnidateArithExpr
						   | MINUS AnnidateArithExpr
		'''
        if len(p) > 3:
            oper = Operation(p[1].value, str(p[2]), p[3].value)
            node = self.manager.createNode(oper, [p[1], p[3]])
            p[0] = node
        else:
            tmpNode = self.manager.createNode(Number("0"), [])
            oper = Operation(tmpNode.value, str(p[1]), p[2].value)
            p[0] = self.manager.createNode(oper, [tmpNode, p[2]])
        self.myPrint("BinaryArithExpr", p)

    def p_UnaryArithExp(self, p):
        '''AnnidateArithExpr : EXP LPAREN AnnidateArithExpr RPAREN
							 | EXP LPAREN BinaryArithExpr RPAREN
		'''

        oper = UnaryOperation(p[3].value, "exp")
        node = self.manager.createNode(oper, [p[3]])
        p[0] = node
        self.myPrint("UnaryArithOp-Exp", p)

    def p_UnaryArithCos(self, p):
        '''AnnidateArithExpr : COS LPAREN AnnidateArithExpr RPAREN
							 | COS LPAREN BinaryArithExpr RPAREN
		'''
        oper = UnaryOperation(p[3].value, "cos")
        node = self.manager.createNode(oper, [p[3]])
        p[0] = node
        self.myPrint("UnaryArithOp-Cos", p)

    def p_UnaryArithSin(self, p):
        '''AnnidateArithExpr : SIN LPAREN AnnidateArithExpr RPAREN
							 | SIN LPAREN BinaryArithExpr RPAREN
		'''
        oper = UnaryOperation(p[3].value, "sin")
        node = self.manager.createNode(oper, [p[3]])
        p[0] = node
        self.myPrint("UnaryArithOp-Sin", p)

    def p_AnnidateArithExpr(self, p):
        '''AnnidateArithExpr : LPAREN AnnidateArithExpr PLUS  AnnidateArithExpr RPAREN
							 | LPAREN AnnidateArithExpr MINUS AnnidateArithExpr RPAREN
							 | LPAREN AnnidateArithExpr MUL AnnidateArithExpr RPAREN
							 | LPAREN AnnidateArithExpr DIVIDE AnnidateArithExpr RPAREN
							 | LPAREN MINUS AnnidateArithExpr RPAREN
		'''

        if len(p) > 5:
            oper = Operation(p[2].value, str(p[3]), p[4].value)
            node = self.manager.createNode(oper, [p[2], p[4]])
            p[0] = node
        else:
            tmpNode = self.manager.createNode(Number("0.0"), [])
            oper = Operation(tmpNode.value, str(p[2]), p[3].value)
            p[0] = self.manager.createNode(oper, [tmpNode, p[3]])
        self.myPrint("AnnidateArithExpr", p)

    def p_UnaryExpr(self, p):
        '''AnnidateArithExpr : UnaryExpr
							 |  LPAREN UnaryExpr RPAREN
		'''

        if len(p) > 2:
            p[0] = p[2]
        else:
            p[0] = p[1]
        self.myPrint("UnaryArithExpr", p)

    def p_Number(self, p):
        '''UnaryExpr : POSNUMBER'''
        p[0] = self.manager.createNode(Number(str(p[1])), [])
        self.myPrint("Number", p)

    def p_Variable(self, p):
        '''UnaryExpr : WORD '''
        if not str(p[1]) in self.variables:
            print("Variable: " + str(p[1]) + "not declared!")
            exit(-1)
        else:
            p[0] = self.manager.createNode(self.variables[str(p[1])], [])
        self.myPrint("Variable", p)

    def p_error(self, p):
        if p:
            raise Exception(
                "Syntax error at '%s', type %s, on line %d, program '%s'" % (p.value, p.type, p.lineno, self.program))
            exit(-1)
        else:
            raise Exception("Syntax error at EOF, program '%s'", self.program)
            exit(-1)
