from project_utils import setCurrentContextPrecision, printMPFRExactly, resetContextDefault

from gmpy2 import mpfr
from pacal import ConstDistr

class ErrorModelPointMass:
    def __init__(self, wrapperInputDistribution, precision, exponent):
        self.wrapperInputDistribution = wrapperInputDistribution
        self.inputdistribution = self.wrapperInputDistribution.execute()
        self.inputdistribution.get_piecewise_pdf()
        self.precision = precision
        self.exponent = exponent
        self.unit_roundoff = 2 ** -self.precision
        setCurrentContextPrecision(self.precision, self.exponent)
        qValue = printMPFRExactly(mpfr(str(self.inputdistribution.rand(1)[0])))
        resetContextDefault()
        error = float(str(self.inputdistribution.rand(1)[0])) - float(qValue)
        self.distribution = ConstDistr(float(error))
        self.distribution.get_piecewise_pdf()

    def execute(self):
        return self.distribution
