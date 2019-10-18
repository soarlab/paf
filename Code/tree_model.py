from fpryacc import FPRyacc
from error_model import ErrorModel

class BinaryTree(object):
    def __init__(self, name, root, left=None, right=None):
        self.name = name
        self.root = root
        self.left = left
        self.right = right

class TreeModel:
    def __init__(self, my_yacc, precision, exp, poly_precision, initialize=False):
        self.initialize = initialize
        self.yacc = my_yacc
        self.precision = precision
        self.exp = exp
        self.poly_precision = poly_precision
        self.tree = self.evaluate(my_yacc.expression)

    # Recursively populate the Tree with the triples (distribution, error distribution, quantized distribution)
    def evaluate(self, node):
        triple = []
        # Test if we're at a leaf
        if node.leaf:
            # Non-quantized distribution
            dist = node.value
            # initialize=True means we quantize the inputs
            if self.initialize:
                # Compute error model
                error = ErrorModel(dist, self.precision, self.exp, self.poly_precision)
                quantized_distribution = dist*(1+error)
            # Else we leave the leaf distribution unchanged
            else:
                error = 0
                quantized_distribution = dist
        # If not at a leaf we need to get the distribution and quantized distributions of the children nodes.
        # Then, check the operation. For each operation the template is the same:
        # dist will be the non-quantized operation the non-quantized children nodes
        # qdist will be the non-quantized operation on the quantized children nodes
        # quantized_distribution will be the quantized operation on the quantized children nodes
        else:
            left = self.evaluate(node.children[0])
            right = self.evaluate(node.children[1])
            if node.value.operator == "*":
                dist = left.root[0]*right.root[0]
                qdist = left.root[2]*right.root[2]
                error = ErrorModel(qdist, self.precision, self.exp, self.poly_precision)
                quantized_distribution = qdist*(1+error)
        # We now populate the triple with distribution, error model, quantized distribution
        triple.append(dist)
        triple.append(error)
        triple.append(quantized_distribution)
        name = node.value.name
        return(BinaryTree(name, triple))


