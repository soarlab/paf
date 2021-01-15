from decimal import Decimal

from SymbolicAffineArithmetic import CreateSymbolicErrorForDistributions, CreateSymbolicErrorForErrors, \
    SymbolicAffineInstance, SymExpression, CreateSymbolicZero
from error_model import HighPrecisionErrorModel, LowPrecisionErrorModel, TypicalErrorModel, FastTypicalErrorModel
from mixedarithmetic import createDSIfromDistribution, createAffineErrorForLeaf
from project_utils import dec2Str
from setup_utils import discretization_points, digits_for_range
from model import BoundingPair


class ErrorModelWrapper:
    """
    Wrapper class implementing only the methods which are required from tree_model
    Input: an ErrorModel object and optionally the corresponding input distribution (to construct standardized name)
    """

    def __init__(self, error_model):
        self.distribution = error_model
        self.sampleInit = True
        self.unit_roundoff = error_model.unit_roundoff
        self.name = error_model.getName()
        self.discretization = None
        self.affine_error = None
        self.symbolic_error = None
        self.bounding_pair = BoundingPair()
        self.get_discretization()
        self.bounding_pair.instantiate_from_distribution(error_model)

    def __str__(self):
        return self.name

    def execute(self):
        return self.distribution

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n)
            self.sampleInit = False
        return self.sampleSet

    def get_discretization(self):
        if self.discretization==None and self.affine_error==None and self.symbolic_error==None:
            tmp_error=self.distribution*self.unit_roundoff
            tmp_error.name=self.name
            tmp_error.a_real=("-{0:."+str(digits_for_range)+"f}").format(self.unit_roundoff)
            tmp_error.b_real=("{0:."+str(digits_for_range)+"f}").format(self.unit_roundoff)
            self.discretization = createDSIfromDistribution(tmp_error, n=discretization_points)
            self.affine_error= createAffineErrorForLeaf()
            self.symbolic_error = CreateSymbolicZero()
            #self.symbolic_affine = CreateSymbolicErrorForErrors(eps_symbol=dec2Str(Decimal(self.unit_roundoff)))
            self.symbolic_affine = CreateSymbolicErrorForErrors(eps_symbol="eps",
                                                                eps_value_string=dec2Str(Decimal(self.unit_roundoff)))
        return self.discretization

