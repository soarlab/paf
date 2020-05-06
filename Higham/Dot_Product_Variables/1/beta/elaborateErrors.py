import os
import numpy as np

quartiles=["99.99%"]

#fptaylor=open("./absfpt.py.txt","r").read()
#vals= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 25, 27, 28, 31, 33, 35, 38, 40, 43, 46, 50, 53, 57, 61, 66, 71, 76, 81, 87, 93, 100]

vals= [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,   14,   17,   21,   24,   27,   31,   34,
        37,   41,   44,   47,   51,   54,   57,   61,   64,   67,   71,
        74,   77,   81,   84,   87,   91,   94,   97,  101,  104,  107,
        111,  114,  117,  121,  124,  127,  131,  134,  137,  141,  144,
        148,  151,  154,  158,  161,  164,  168,  171,  174,  178,  181,
        184,  188,  191,  194,  198,  201]#,  204]

'''        
        ,  208,  211,  214,  218,
        221,  224,  228,  231,  234,  238,  241,  244,  248,  251,  254,
        258,  261,  264,  268,  271,  274,  278,  281,  284,  288,  291,
        295,  298,  301,  305,  308,  311]

       ,  315,  318,  321,  325,  328,
        331,  335,  338,  341,  345,  348,  351,  355,  358,  361,  365,
        368,  371,  375,  378,  381,  385,  388,  391,  395,  398,  401,
        405,  408,  411,  415,  418,  421,  425,  428,  432,  435,  438,
        442,  445,  448,  452,  455,  458,  462,  465,  468,  472,  475, 478, 480]
'''
res=[]

for num in vals:
	tmp=[]
	folder="test_b(2,2)_"+str(num)
	
	#print("\n\n\n\n\n"+folder)
	#golden=open("./"+folder+"/golden.txt","r").read()
	#sampling=open("./"+folder+"/sampling.txt","r").read()
	
	FPPM=open("./"+folder+"/"+folder+"_CDF_summary.out","r").read()
		
	lookFor="CDF Rel Error Analysis"
	
	#for line in fptaylor.split("\n"):
	#	if folder in line:
	#		fptaylor_vals=float(line.split()[1]) 
	#		print("FPtaylor: "+("{:.2e}".format(fptaylor_vals))+"\n")
	#		break
	
	for quart in quartiles:
		
		#golden_range=(golden.split(lookFor)[1]).split("##################################")[0]
		#sampling_range=(sampling.split(lookFor)[1]).split("##################################")[0]
		FPPM_range=(FPPM.split(lookFor)[1]).split("##################################")[0]
		
		for line in FPPM_range.split("\n"):
			if quart in line:
				FPPM_vals=float((((line.split("[")[1]).split("]")[0]).split(","))[1])
				#print(("{:.7e}".format(FPPM_vals)))
				print(float("{:.7e}".format(FPPM_vals)))
