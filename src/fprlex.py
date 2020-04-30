import sys

import ply.lex as lex

if sys.version_info[0] >= 3:
    raw_input = input

class FPRlex(object):
	def __init__(self):
		# Build the lexer
		self.lexer = lex.lex(module=self)
	
	reserved = {
		'N' : 'N',
		'B' : 'B',
		'U' : 'U',
		'if': 'if',
		'exp' : 'EXP',
		'cos': 'COS',
		'sin': 'SIN'
	}
	
	tokens = list(dict.fromkeys([
		'POSNUMBER',
		'WORD',
		'PLUS',
		'MINUS',
		'MUL',
		'DIVIDE',
		'COMMA',
		'EXP',
		'SIN',
		'COS',
		'LPAREN',
		'RPAREN',
		'NEWLINE',
		'COLON'] + list(reserved.values())))
		
	# Regular expression rules for simple tokens
	t_PLUS    = r'\+'
	t_MINUS   = r'-'
	t_MUL   = r'\*'
	t_DIVIDE  = r'/'
	t_LPAREN  = r'\('
	t_RPAREN  = r'\)'
	t_COMMA = r','
	t_COLON = r':'


	def t_POSNUMBER(self,t):
		r'([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
		t.value=str(t.value)
		return t

	'''
	def t_NEGNUMBER(self,t):
		r'(-[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
		t.value=str(t.value)
		return t
	'''

	def t_WORD(self,t):
		r'[a-zA-Z$_][a-zA-Z0-9$_]*'
		t.type = self.reserved.get(t.value,'WORD')
		t.value = str(t.value)
		return t
	
	# Define a rule so we can track line numbers
	def t_NEWLINE(self,t):
		r'\n'
		t.lexer.lineno += len(t.value)
		t.value = str(t.value)
		return t
	
	def t_COMMENT(self,t):
		r'\%.*'
		pass
		# No return value. Token discarded

	# Error handling rule
	def t_error(self,t):
		print("Found Illegal character "+t.value[0])
		exit(0)
		 
	# A string containing ignored characters (spaces and tabs)
	t_ignore  = ' \t\r'
