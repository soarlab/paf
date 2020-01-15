import os
for myfile in os.listdir("./"):
	if myfile.endswith(".txt"):
		os.rename(myfile, myfile.split(".")[0]+"_gaussian.txt")
