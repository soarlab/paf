
from pychebfun import *

class ConditionalError:
    """
    A class of the evaluation of the relative error distribution of
    an expression (passed as a tree_model)
    using Monte-Carlo integration with sample_nb samples at the leaves of the tree0
    and cheb_point_nb interpolation points to construct the final distribution
    """
    def __init__(self, expression, sample_nb, cheb_point_nb, precision):
        self.expression = expression
        self.sample_nb = sample_nb
        self.cheb_point_nb = cheb_point_nb
        self.unit_roundoff = 2 ** (-precision)
        # initialize node number
        self.error_range = 0
        # compute error range (in units of u)
        self.get_error_range(expression.tree)
        self.interpolation_points = Chebfun.interpolation_points(self.cheb_point_nb)
        for i in range(0, self.cheb_point_nb):
            self.interpolation_points[i] *= (self.unit_roundoff * self.error_range)

    def get_error_range(self, tree):
        """
        Compute range of the relative error as
        +/- the number of non-leaf nodes times u (if no initial quantization)
        +/- the number of nodes times u (with initial quantization)
        """
        if tree.left is not None or tree.right is not None:
            self.error_range += 1
            if tree.left is not None:
                self.get_error_range(tree.left)
            if tree.right is not None:
                self.get_error_range(tree.right)
        else:
            if tree.root_value[1] is not 0:
                self.error_range += 1

    def get_monte_carlo_error(self):
        d = np.zeros([self.sample_nb,self.cheb_point_nb])
        d_final = np.zeros(self.cheb_point_nb)
        for i in range(0, self.sample_nb):
            exact_at_sample, error_at_sample=self.get_error_at_sample(self.expression.tree)
            relative_error = (exact_at_sample - error_at_sample) / exact_at_sample
            # evaluate at cheb_point_nb
            for j in range(0, self.cheb_point_nb):
                d[i,j]=relative_error.get_piecewise_pdf()(self.interpolation_points[j])
        d_final = np.sum(d, axis=0)
        d_final = d_final / (self.sample_nb * self.unit_roundoff)
        return d_final

    def get_error_at_sample(self, tree):
        """
        Given a TreeModel tree this method does the following:
        1) it goes to the leaves of the tree and sample each distribution there
        2) it then recursively navigates up to tree to compute:
            a) the exact value of the expression where each variable is instantiated to its sampled value
            b) the output distribution where each variable is instantiated to its sampled value
            and each error distribution is taken from the TreeModel
        """
        if tree.left is not None or tree.right is not None:
            if tree.left is not None:
                exact_l, prob_l = self.get_error_at_sample(tree.left)
            if tree.right is not None:
                exact_r, prob_r = self.get_error_at_sample(tree.right)
            if tree.root_name == "+":
                return (exact_l + exact_r), (prob_l + prob_r) * (1 + tree.root_value[1].distribution * self.unit_roundoff)
            elif tree.root_name == "-":
                return (exact_l - exact_r), (prob_l - prob_r) * (1 + tree.root_value[1].distribution * self.unit_roundoff)
            elif tree.root_name == "*":
                return (exact_l * exact_r), (prob_l * prob_r) * (1 + tree.root_value[1].distribution * self.unit_roundoff)
            elif tree.root_name == "/":
                return (exact_l / exact_r), (prob_l / prob_r) * (1 + tree.root_value[1].distribution * self.unit_roundoff)
            else:
                print("Operation not supported!")
                exit(-1)
        else:
            sample = tree.root_value[0].execute().rand()
            return sample, sample
