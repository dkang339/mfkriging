import numpy as np

def relerr(ytrue,pred):

    # relative error
    ytrue = ytrue.ravel()
    pred = pred.ravel()
    err = np.abs((ytrue-pred)/ytrue)
    meanerr = np.mean(err)

    return meanerr