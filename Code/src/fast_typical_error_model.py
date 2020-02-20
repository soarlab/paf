from typical_error_model import TypicalErrorModel

from pacal.segments import PiecewiseDistribution, Segment
from pacal.utils import wrap_pdf

###
# Approximate Error Model given by the "Typical Distribution"
###
class FastTypicalErrorModel(TypicalErrorModel):
    """
    An implementation of the (fast) typical error distribution with three segments
    """
    def __init__(self, input_distribution=None, precision=None, **kwargs):
        super(FastTypicalErrorModel, self).__init__(input_distribution, precision)
        self.name = "FastTypicalErrorDistribution"

    def init_piecewise_pdf(self):
        piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        piecewise_pdf.addSegment(Segment(-1, 1, wrapped_pdf))
        #piecewise_pdf.addSegment(Segment(-0.5, 0.5, wrapped_pdf))
        #piecewise_pdf.addSegment(Segment(0.5, 1, wrapped_pdf))
        self.piecewise_pdf = piecewise_pdf.toInterpolated()
