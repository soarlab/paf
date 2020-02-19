
class ErrorModelWrapper:
    """
    Wrapper class implementing only the methods which are required from tree_model
    Input: an ErrorModel object
    """
    def __init__(self, error_model, precision, exp):
        self.precision=precision
        self.exp=exp
        self.eps = 2 ** (-self.precision)
        self.error_model = error_model
        self.sampleInit = True
        self.name=self.getName()

    def __str__(self):
        return self.error_model.getName()

    def getName(self):
        return self.error_model.getName()

    def execute(self):
        return self.error_model.execute()

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.error_model.rand(n)
            self.sampleInit = False
        return self.sampleSet


###
# Functions switching between low and double precision. Beware mantissa = gmpy2 precision (includes sign bit)
###
#def setCurrentContextPrecision(mantissa, exponent):
#    ctx = gmpy2.get_context()
#    if mantissa is None:
#        ctx.precision = 24
#    else:
#        ctx.precision = mantissa
#    if exponent is None:
#        ctx.emax = 2 ** 7
#    else:
#        ctx.emax = 2 ** (exponent - 1)
#    ctx.emin = 1 - ctx.emax

#def resetContextDefault():
#    gmpy2.set_context(gmpy2.context())
###
# Wrapper class for all Error Models
###

###
# OLD CODE
###

# my_pdf = None
#
#
# def genericPdf(x):
#     if isinstance(x, float) or isinstance(x, int) or len(x) == 1:
#         if x < -1 or x > 1:
#             return 0
#         else:
#             return my_pdf(x)
#     else:
#         res = np.zeros(len(x))
#         for index, ti in enumerate(x):
#             if ti < -1 or ti > 1:
#                 res[index] = 0
#             else:
#                 res[index] = my_pdf(ti)
#         return res
#     exit(-1)
#
# def getTypical(x):
#     if isinstance(x, float) or isinstance(x, int) or len(x) == 1:
#         if abs(x) <= 0.5:
#             return 0.75
#         else:
#             return 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
#     else:
#         res = np.zeros(len(x))
#         for index, ti in enumerate(x):
#             if abs(ti) <= 0.5:
#                 res[index] = 0.75
#             else:
#                 res[index] = 0.5 * ((1.0 / ti) - 1.0) + 0.25 * (((1.0 / ti) - 1.0) ** 2)
#         return res
#     exit(-1)
#
#
# typVariable = None
#
#
# def createTypical(x):
#     return typVariable(x)


# class WrappedPiecewiseTypicalError():
#
#     def __init__(self, p=None):
#         self.sampleInit = True
#         self.distribution = PiecewiseTypicalError(p)
#         self.distribution.init_piecewise_pdf()
#
#     def execute(self):
#         return self.distribution
#
#     def getSampleSet(self, n=100000):
#         # it remembers values for future operations
#         if self.sampleInit:
#             self.sampleSet = self.distribution.rand(n)
#             self.sampleInit = False
#         return self.sampleSet


# class TypicalErrorModel:
#     def __init__(self, precision, exp, poly_precision):
#         self.poly_precision = poly_precision
#         self.name = "E"
#         self.precision = precision
#         self.exp = exp
#         self.sampleInit = True
#         self.unit_roundoff = 2 ** (-self.precision)
#         self.distribution = self.createTypicalErrorDistr()
#
#     def createTypicalErrorDistr(self):
#         global typVariable
#         typVariable = chebfun(getTypical, domain=[-1.0, 1.0], N=self.poly_precision)
#         self.distribution = FunDistr(createTypical, breakPoints=[-1.0, 1.0], interpolated=True)
#         self.distribution.get_piecewise_pdf()
#         return self.distribution
#
#     def execute(self):
#         return self.distribution
#
#     def getSampleSet(self, n=100000):
#         # it remembers values for future operations
#         if self.sampleInit:
#             self.sampleSet = self.distribution.rand(n)
#             self.sampleInit = False
#         return self.sampleSet
#
#
#
