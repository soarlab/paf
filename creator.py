input_file=open("./test.txt","w+")

declarations=""
for i in range(0,5):
	declarations=declarations+"x"+str(i)+":B(2.0,2.0), "+"y"+str(i)+":B(2.0,2.0), "

declarations=declarations[:-2]+"\n"

expr="(x0*y0)"
for i in range(1,5):
	expr="("+expr+"+(x"+str(i)+"*y"+str(i)+")"+")"

final=declarations+expr
input_file.write(final+"\n")
input_file.close()
