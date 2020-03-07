import numpy as np

def error_free_scalar_product(X, Y):
    #Test if X,Y are scalars or arrays
    if np.isscalar(X):
        XX=[]
        XX.append(X)
    else:
        XX=X
    if np.isscalar(Y):
        YY=[]
        YY.append(Y)
    else:
        YY=Y

    # Test if X,Y have the same dimension
    if len(X)!=len(Y):
        raise Exception('The input vectors must have the same dimensions')

    i=0
    for Xi in XX:
        SP=SP+(Xi*YY[i])
        print(i)
        print(Xi.summary())
        print(YY[i].summary())
        print(SP.summary())
        i+=1

    return SP
