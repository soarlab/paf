from error_model import HighPrecisionErrorModel, LowPrecisionErrorModel, TypicalErrorModel, FastTypicalErrorModel


class ErrorModelWrapper:
    """
    Wrapper class implementing only the methods which are required from tree_model
    Input: an ErrorModel object and optionally the corresponding input distribution (to construct standardized name)
    """

    def __init__(self, error_model, input_distribution_wrapper=None):
        self.distribution = error_model
        self.sampleInit = True
        self.unit_roundoff = error_model.unit_roundoff
        if input_distribution_wrapper is not None:
            self.input_name = input_distribution_wrapper.name
        self.name = self.getName()

    def __str__(self):
        return self.name

    def getName(self):
        if isinstance(self.distribution, HighPrecisionErrorModel):
            name = "HPError(" + self.input_name + ")"
        elif isinstance(self.distribution, TypicalErrorModel):
            name = "TypicalError"
        elif isinstance(self.distribution, FastTypicalErrorModel):
            name = "TypicalError"
        elif isinstance(self.distribution, LowPrecisionErrorModel):
            name = "LPError(" + self.input_name + ")"
        return name

    def execute(self):
        return self.distribution

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n)
            self.sampleInit = False
        return self.sampleSet