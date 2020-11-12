from error_model import HighPrecisionErrorModel, LowPrecisionErrorModel, TypicalErrorModel, FastTypicalErrorModel
from mixedarithmetic import createDSIfromDistribution, createAffineErrorFromDistribution
from setup_utils import discretization_points


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
        self.discretization=None
        self.affine_error = None
        self.get_discretization()

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
        if self.discretization==None and self.affine_error==None:
            self.discretization = createDSIfromDistribution(self.distribution*self.unit_roundoff, n=discretization_points)
            self.affine_error= createAffineErrorFromDistribution()
        return self.discretization

