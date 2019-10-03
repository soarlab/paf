import pacal
import pychebfun as cheb
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


def regularizeDistribution(D):
    approxLimit = 1000 * np.finfo(np.float32).eps
    f = D.get_piecewise_pdf()
    newSegments = [f.segments[0]]
    i = 1
    while i < len(f.segments) - 1:
        j = i
        # Identify a sequence of standard InterpolatedSegments without large discontinuities or gradient changes
        while smoothnessCriterion(f.segments[j], f.segments[j + 1]):
            j += 1
        # Segments i to j can be merged
        merged = mergeSegments(f, i, j)
        # Quality control: is the new segment a good approximation of segments i to j
        if i < j:
            if computeDistance(merged, f, f.segments[i].a, f.segments[j].b) < approxLimit:
                newSegments.append(merged)
            # Quality control has failed, keep all segments
            else:
                for k in range(i, j):
                    newSegments.append(f.segments[k])
        else:
            newSegments.append(merged)
        i = j + 1
    newSegments.append(f.segments[i])
    breakPoints = generateBreakPoints(newSegments)
    newD = pacal.FunDistr(lambda ti: wrapSegments(newSegments,ti),breakPoints)
    newD.init_piecewise_pdf()
    #plt.close("all")
    #plt.figure("test")
    #newD.plot()
    return newD


# Tests if two CONSECUTIVE segments s,t can be merged into one.
# Returns a boolean
def smoothnessCriterion(s, t):
    # Constants used to test for smoothness
    jumpLimit = np.finfo(np.float32).eps
    deltavLimit = 10 * np.finfo(np.float32).eps
    # Test for non-standard types of segments
    if type(s.f).__name__ is "ChebyshevInterpolator" and type(t.f).__name__ is "ChebyshevInterpolator":
        # Test for discontinuities between segments
        if abs(s.f(s.b) - t.f(t.a)) < jumpLimit:
            if abs(s.f.diff()(s.b) - t.f.diff()(t.a)) < deltavLimit:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def mergeSegments(f, i, j):
    if i is j:
        return f.segments[i]
    else:
        # Range of the new segment
        a = f.segments[i].a
        b = f.segments[j].b
        interp=cheb.chebfun(f, domain=[a, b], N=100)
        tmp=pacal.segments.InterpolatedSegment(a,b,interp)
        return tmp


def restrictRange(f, t, a, b):
    y = []
    for x in t:
        if a < x < b:
            y.append(f(x))
        else:
            y.append(0)


def computeDistance(segment, f, a, b, p=1):
    result = integrate.quad(lambda x: norm(segment.f(x) - f(x), p), a, b)
    return result[0]


def norm(x, p=1):
    if p is 1:
        return abs(x)
    elif p > 1:
        return pow(x, p)
    else:
        raise ValueError("Invalid norm parameter")

def wrapSegments(segments, t):
    #if ti < segments[0].a or ti > segments[len(segments)-1].b:
    #    raise ValueError("t is out of bounds")
    if isinstance(t,float) or isinstance(t,int) or len(t)==1:
        for s in segments:
            if t <= s.b:
                return s.f(t)
    else:
        res=np.zeros(len(t))
        tis=t
        for index,ti in enumerate(tis):
            for s in segments:
                if ti <= s.b:
                    res[index]=s.f(ti).item(0)
                    break
        return res

def generateBreakPoints(segments):
    bp=[]
    for s in segments:
        if type(s).__name__ is "Chebfun":
            bp.append(s._domain.min())
        else:
            bp.append(s.a)
    if type(s).__name__ is "Chebfun":
        bp.append(s._domain.max())
    else:
        bp.append(s.b)
    return bp

def testRegularizer():
    plt.close("all")
    X = pacal.NormalDistr(6, 0.1)
    Y = pacal.BetaDistr(5, 7)
    Z = pacal.NormalDistr(-5, 0.1)
    T = (X + Y) + Z
    plt.figure("Original Distribution")
    T.plot()
    rT = regularizeDistribution(T)
    plt.figure("New Distribution")
    rT.plot()
    print ("Done")