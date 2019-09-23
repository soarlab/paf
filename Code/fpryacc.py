import sys
from __builtin__ import str

import ply.yacc as yacc
from fprlex import *
from model import *
import fprlex

uniqueLexer=FPRlex()
			
class FPRyacc:
	
	tokens = FPRlex.tokens
	reserved = FPRlex.reserved

	def __init__(self,program,debug):
		self.lexer = uniqueLexer
		self.parser = yacc.yacc(module=self)
		self.debug = debug
		self.program = program
		self.variables = {}
		self.expression=self.parser.parse(self.program)

	def addVariable(self,name,distribution):
		if name in self.variables:
			print "Variable: " + name + " declared twice!"
			exit(-1)
		else:
			self.variables[name]=distribution

	def myPrint(self,s,p):
		res=""
		if self.debug:
			print str(s)+": "+str(p[0])
		return
	
	def p_FileInput(self, p):
		''' FileInput : VarDeclaration NEWLINE Expression
		'''
		p[0]=p[3]
		self.myPrint("FileInput",p)
		
	def p_VarDeclaration(self, p):
		''' VarDeclaration : Distribution
						   | Distribution COMMA VarDeclaration
		'''
		#if len(p)>2:
		#	p[0]=str(p[1])+str(p[2])+str(p[3])
		#elif len(p)==2:
		#	p[0]=str(p[1])
		self.myPrint("VarDeclaration",p)
		
	def p_Uniform(self, p):
		''' Distribution : WORD COLON U LPAREN NUMBER COMMA NUMBER RPAREN
		'''

		distr=U(str(p[1]),str(p[5]),str(p[7]))
		self.addVariable(str(p[1]),distr)
		#p[0] = distr
		self.myPrint("Uniform",p)

	def p_Normal(self, p):
		''' Distribution : WORD COLON N LPAREN NUMBER COMMA NUMBER RPAREN
		'''

		#p[0] = str(p[1]) + str(p[2]) + str(p[3]) + str(p[4]) + str(p[5]) + str(p[6]) + str(p[7]) + str(p[8])
		distr = N(str(p[1]), str(p[5]), str(p[7]))
		self.addVariable(str(p[1]),distr)
		#p[0] = distr
		self.myPrint("Normal", p)

	def p_Beta(self, p):
		''' Distribution : WORD COLON B LPAREN NUMBER COMMA NUMBER RPAREN
		'''
		#p[0] = str(p[1]) + str(p[2]) + str(p[3]) + str(p[4]) + str(p[5]) + str(p[6]) + str(p[7]) + str(p[8])
		distr = B(str(p[1]), str(p[5]), str(p[7]))
		self.addVariable(str(p[1]),distr)
		#p[0] = distr
		self.myPrint("Beta", p)

	def p_Expression(self, p):
		'''Expression : AnnidateArithExpr
					  | BinaryArithExpr
		'''
		p[0] = p[1]
		self.myPrint("Expression",p)
	
	def p_BinaryArithExpr(self, p):
		'''BinaryArithExpr : AnnidateArithExpr PLUS  AnnidateArithExpr
						   | AnnidateArithExpr MINUS AnnidateArithExpr
						   | AnnidateArithExpr MUL AnnidateArithExpr
						   | AnnidateArithExpr DIVIDE AnnidateArithExpr
						   | MINUS AnnidateArithExpr
		'''
		if len(p)>3:
			p[0]=Node(Operation(p[1].value,str(p[2]),p[3].value,False),[p[1],p[3]])
		else:
			tmp=Node(Number("0"))
			p[0]=Node(Operation(tmp.value,str(p[1]),p[2].value,False),[tmp,p[2]])
		self.myPrint("BinaryArithExpr",p)
		
	def p_AnnidateArithExpr(self, p):
		'''AnnidateArithExpr : LPAREN AnnidateArithExpr PLUS  AnnidateArithExpr RPAREN
							 | LPAREN AnnidateArithExpr MINUS AnnidateArithExpr RPAREN
							 | LPAREN AnnidateArithExpr MUL AnnidateArithExpr RPAREN
							 | LPAREN AnnidateArithExpr DIVIDE AnnidateArithExpr RPAREN
							 | LPAREN MINUS AnnidateArithExpr RPAREN
		'''

		if len(p)>5:
			p[0]=Node(Operation(p[2].value,str(p[3]),p[4].value,True),[p[2],p[4]])
		else:
			tmp=Node(Number("0"))
			p[0]=Node(Operation(tmp.value,str(p[3]),p[4].value,True),[tmp,p[3]])
		self.myPrint("AnnidateArithExpr",p)
	
	def p_UnaryExpr(self, p):
		'''AnnidateArithExpr : UnaryExpr
							 |  LPAREN UnaryExpr RPAREN
		'''

		if len(p)>2:
			p[0]=p[2]
		else:
			p[0]=p[1]
		self.myPrint("UnaryArithExpr",p)
	
	def p_Number(self, p):
		'''UnaryExpr : NUMBER'''
		p[0]=Node(Number(str(p[1])))
		self.myPrint("Number",p)
	
	def p_Variable(self, p):
		'''UnaryExpr : WORD '''
		if not str(p[1]) in self.variables:
			print "Variable: " + str(p[1]) + "not declared!"
			exit(-1)
		else:
			p[0]=Node(self.variables[str(p[1])])
		self.myPrint("Variable",p)

	def p_error(self, p):
		if p:
			raise Exception("Syntax error at '%s', type %s, on line %d, program '%s'" % (p.value, p.type, p.lineno, self.program))
			exit(-1)
		else:
			raise Exception("Syntax error at EOF, program '%s'", self.program)
			exit(-1)

	# def p_MinAnnidateArithExpr(self, p):
	# 	'''AnnidateArithExpr : MIN LPAREN AnnidateArithExpr COMMA AnnidateArithExpr RPAREN
	# 	'''
	#
	# 	if not self.buildModelForVarsOnlyForBNB:
	# 		varname=self.solver.encodeMinProblem(str(p[3]),str(p[5]))
	# 		p[0]=varname
	# 	self.myPrint("MinAnnidateArithExpr",p)
	#
	# def p_MaxAnnidateArithExpr(self, p):
	# 	'''AnnidateArithExpr : MAX LPAREN AnnidateArithExpr COMMA AnnidateArithExpr RPAREN
	# 	'''
	#
	# 	if not self.buildModelForVarsOnlyForBNB:
	# 		varname=self.solver.encodeMaxProblem(str(p[3]),str(p[5]))
	# 		p[0]=varname
	# 	self.myPrint("MaxAnnidateArithExpr",p)