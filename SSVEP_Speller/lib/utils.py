import numpy as np
import warnings
import math


def ITR(n, p, t):
    if p < 0 or 1 < p:
        raise Exception('Accuracy need to be between 0 and 1.')
    elif p < 1 / n:
        warnings.warn('The ITR might be incorrect because the accuracy < chance level.')
        return 0
    elif p == 1:
        return math.log2(n) * 60 / t
    else:
        return (math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))) * 60 / t
