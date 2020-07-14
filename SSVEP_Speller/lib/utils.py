import numpy as np
import scipy.signal as signal
import warnings
import math

PASSBAND = [7.8, 15.6, 23.4, 32.2, 39, 46.8, 54.6, 62.4, 70.2, 78]
STOPBAND = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]


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


def filterband(eeg, band_id, fs, axis=-2):
    fs /= 2
    Wp = [PASSBAND[band_id] / fs, 93 / fs]
    Ws = [STOPBAND[band_id] / fs, 100 / fs]
    N, Wn = signal.cheb1ord(Wp, Ws, 3, 40)
    b, a = signal.cheby1(N, 0.5, Wn, btype="bandpass")
    return signal.filtfilt(b, a, eeg, axis=axis)