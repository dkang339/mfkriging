from mfmodel.mfkriging import MFKriging
import numpy as np
import h5py
from postprocess import relerr
regularize = True

def hikrig(model,n,m,k,split=False):

    # define variables
    with h5py.File(model["fname_h"], "r") as f:
        h_out = f["output"][:] # (n,)
        input = f["input"][:] # (n_data,d_in)
    with h5py.File(model["fname_l"], "r") as f:
        l_out = f["output"][:] # (m,)
    data = {
    "input": input,
    "h_out": h_out,
    "l_out": l_out
}
    n_input = len(data["input"])

    if split: # split highfi data into training and testing sets
        n_test_max = 500
        # setup testing data
        rng = np.random.default_rng(42)
        test_idx = rng.choice(n_input, n_test_max, replace=False)
        p_test = data["input"][test_idx]
        h_test = data["h_out"][test_idx]
        h_train_idx = np.setdiff1d(np.arange(n_input), test_idx) # exclude testing indices
        l_train_idx = np.arange(n_input) # use all lowfi indices

    else: # if we already have separate testing data
        with h5py.File(model["fname_test"], "r") as f:
            p_test = f["input"][:] # (n_test,d_in)
            h_test = f["output"][:] # (n_test,)
        h_train_idx = np.arange(n_input) # use all highfi data for training
        l_train_idx = np.arange(n_input) # use all lowfi indices

    # setup training data
    rng = np.random.default_rng(k)
    h_idx = rng.choice(h_train_idx, n, replace=False) # highfi indices
    # h_idx0 = np.setdiff1d(h_train_idx, h_idx) # exclude the chosen highfi indices
    l_idx = np.setdiff1d(l_train_idx, h_idx) # lowfi indices excluding the chosen highfi indices
    lonly_idx = rng.choice(l_idx, m-n, replace=False) # lowfi indices not in highfi
    p_htrain = data["input"][h_idx] # highfi training inputs (n, d_in)
    p_ltrain = np.concatenate((data["input"][h_idx], data["input"][lonly_idx]), axis=0) # (m, d_in) lowfi training inputs
    h_train = data["h_out"][h_idx] # highfi training outputs (n,)
    l_train = np.concatenate((data["l_out"][h_idx], data["l_out"][lonly_idx]), axis=0) # lowfi training samples
    
    # instantiate MFKriging class
    surrogate = MFKriging(kernel='ardmatern32',
                            regularize=regularize,
                            normalize=True,
                            n_restart=10)
    surrogate.train(p_ltrain, l_train, p_htrain, h_train)

    # predict outputs for testing inputs
    pred = surrogate.predict(p_test, return_var=False) # (n_test,)

    # compute relative error
    err = relerr(h_test, pred)

    return err
