from pacal import *

X = UniformDistr(10, 15.5)
Y = UniformDistr(0.97, 2)
Z = X / Y
Z.plot()
show()