'''
    A simple implementation of the discrete correlation function (DCF)
    Author: Damien Robertson - robertsondamien@gmail.com
    Usage:
      $ python dcf.py -h for help and basic instruction
'''

from __future__ import print_function, division

print(__doc__)

try:
    import numpy as np
except ImportError:
    print("Numpy not installed, try - pip install numpy")
    import sys
    sys.exit()

#
#   Subroutines
#

def tsdtrnd(ts, vrbs, plyft):

    '''
        Subroutine - tsdtrnd
          Time series detrend using the user chosen polynomial order. Subroutine
          fits a ploynomial to the time series data and subtracts.
        Requires scipy.optimize (scipy) to be installed.
    '''

    if plyft == 0:

        ts_mean = np.mean(ts[:,1])
        ts[:,1] = ts[:,1] - ts_mean
        if vrbs:
            print("Mean subtraction: %.4e" %ts_mean)

    elif plyft == 1:

        try:
            from scipy.optimize import curve_fit
        except ImportError:
            print("Scipy not installed, try - pip install scipy")
            import sys
            sys.exit()

        lnfnc = lambda x, a, b: a*x + b
        p0, c0 = curve_fit(lnfnc, ts[:,0], ts[:,1], sigma=ts[:,2])
        ts[:,1] = ts[:,1] - lnfnc(ts[:,0], p0[0], p0[1])

        if vrbs:
            print("Linear De-trend Coefficients [a*x + b]")
            print("a:", p0[0])
            print("b:", p0[1])

    else:

        try:
            from scipy.optimize import curve_fit
        except ImportError:
            print("Scipy not installed, try - pip install scipy")
            import sys
            sys.exit()

        lnfnc = lambda x, a, b, c: a*x**2.0 + b*x + c
        p0, c0 = curve_fit(lnfnc, ts[:,0], ts[:,1], sigma=ts[:,2])
        ts[:,1] = ts[:,1] - lnfnc(ts[:,0], p0[0], p0[1], p0[2])

        if vrbs:
            print("Quadratic De-trend Coefficients [a*x**2 + b*x + c]")
            print("a:", p0[0])
            print("b:", p0[1])
            print("c:", p0[2])

    return ts

def set_unitytime(ts1, ts2):

    '''
        Subroutine - set_unitytime
          Simply shifts both time series so that one starts at zero.
    '''

    unitytime = min(np.min(ts1[:,0]), np.min(ts2[:,0]))
    ts1[:,0] = ts1[:,0] - unitytime
    ts2[:,0] = ts2[:,0] - unitytime

    return ts1, ts2

def chck_tserr(ts):

    '''
        Subroutine - chck_tserr
          Makes sure user has entered a properly formatted ts file.
          Checks to see if input time series has a measurement error column - third
          column of input file.
    '''

    assert ((ts.shape[1] == 2) or (ts.shape[1] == 3)), "TS SHAPE ERROR"

    if ts.shape[1] == 2:
        ts_fill = np.zeros((ts.shape[0], 3))
        ts_fill[:,0:2] = ts[:,0:2]

        return ts_fill

    else:

        return ts

def get_timeseries(infile1, infile2, vrbs, plyft):

    '''
        Subroutine - get_timeseries
          Takes the user specified filenames and runs tsdtrnd and set_unitytime.
          Returns the prepared time series for DCF.
    '''

    ts1_in = np.loadtxt(infile1, comments='#', delimiter=',')
    ts2_in = np.loadtxt(infile2, comments='#', delimiter=',')

    ts1 = chck_tserr(ts1_in)
    ts2 = chck_tserr(ts2_in)

    ts1, ts2 = set_unitytime(ts1, ts2)
    ts1 = tsdtrnd(ts1, vrbs, plyft)
    ts2 = tsdtrnd(ts2, vrbs, plyft)

    return ts1, ts2

def get_timeseries_ipynb(ts1_in, ts2_in, vrbs, plyft):

    '''
        Subroutine - get_timeseries
          Takes the user specified filenames and runs tsdtrnd and set_unitytime.
          Returns the prepared time series for DCF.
    '''

    ts1 = chck_tserr(ts1_in)
    ts2 = chck_tserr(ts2_in)

    ts1, ts2 = set_unitytime(ts1, ts2)
    ts1 = tsdtrnd(ts1, vrbs, plyft)
    ts2 = tsdtrnd(ts2, vrbs, plyft)

    return ts1, ts2

def sdcf(ts1, ts2, t, dt):

    '''
        Subroutine - sdcf
          DCF algorithm with slot weighting
    '''

    dcf = np.zeros(t.shape[0])
    dcferr = np.zeros(t.shape[0])
    n = np.zeros(t.shape[0])

    dst = np.empty((ts1.shape[0], ts2.shape[0]))
    for i in range(ts1.shape[0]):
        for j in range(ts2.shape[0]):
            dst[i,j] = ts2[j,0] - ts1[i,0]

    for k in range(t.shape[0]):
        tlo = t[k] - dt/2.0
        thi = t[k] + dt/2.0
        ts1idx, ts2idx = np.where((dst < thi) & (dst > tlo))

        mts2 = np.mean(ts2[ts2idx,1])
        mts1 = np.mean(ts1[ts1idx,1])
        n[k] = ts1idx.shape[0]

        dcfdnm = np.sqrt((np.var(ts1[ts1idx,1]) - np.mean(ts1[ts1idx,2])**2) \
                         * (np.var(ts2[ts2idx,1]) - np.mean(ts2[ts2idx,2])**2))

        dcfs = (ts2[ts2idx,1] - mts2) * (ts1[ts1idx,1] - mts1) / dcfdnm

        dcf[k] = np.sum(dcfs) / float(n[k])
        dcferr[k] = np.sqrt(np.sum((dcfs - dcf[k])**2)) / float(n[k] - 1)

    return dcf, dcferr

def gdcf(ts1, ts2, t, dt):

    '''
        Subroutine - gdcf
          DCF algorithm with gaussian weighting
    '''

    h = dt/4.0
    gkrn = lambda x: np.exp(-1.0 * np.abs(x)**2 / (2.0 * h**2)) \
           / np.sqrt(2.0 * np.pi * h)
    cntrbt = gkrn(3.290527*h)

    dcf = np.zeros(t.shape[0])
    dcferr = np.zeros(t.shape[0])
    n = np.zeros(t.shape[0])

    dst = np.empty((ts1.shape[0], ts2.shape[0]))
    for i in range(ts1.shape[0]):
        for j in range(ts2.shape[0]):
            dst[i,j] = ts2[j,0] - ts1[i,0]

    for k in range(t.shape[0]):
        gdst = gkrn(dst - t[k])
        ts1idx, ts2idx = np.where(gdst >= cntrbt)

        mts2 = np.mean(ts2[ts2idx,1])
        mts1 = np.mean(ts1[ts1idx,1])
        n[k] = ts1idx.shape[0]

        dcfdnm = np.sqrt((np.var(ts1[ts1idx,1]) - np.mean(ts1[ts1idx,2])**2) \
                         * (np.var(ts2[ts2idx,1]) - np.mean(ts2[ts2idx,2])**2))

        dcfs = (ts2[ts2idx,1] - mts2) * (ts1[ts1idx,1] - mts1) / dcfdnm
        dcf[k] = np.sum(dcfs) / float(n[k])
        dcferr[k] = np.sqrt(np.sum((dcfs - dcf[k])**2)) / float(n[k] - 1)

    return dcf, dcferr
