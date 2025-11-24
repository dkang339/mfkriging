'''
Preprocess Ishigami data.
'''

from numpy.polynomial.legendre import leggauss
import sys
from pathlib import Path
import numpy as np
import os
import h5py
current_dir = Path(__file__).parent # get current directory
root_dir = current_dir.parent.resolve() # get src directory
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / '../src'))
data_dir = '../../data/ishi'
os.makedirs(data_dir, exist_ok=True)

a = 5
b = 0.1
f1 = lambda x: np.sin(x[:,0]) + a*np.sin(x[:,1])**2 + b*x[:,2]**4*np.sin(x[:,0]) # high fidelity
f2 = lambda x: np.sin(x[:,0]) + 0.6*a*np.sin(x[:,1])**2 + 9*b*x[:,2]**2*np.sin(x[:,0]) # low fidelity

def stats(npts=50):
    ''''
    compute stats for MFMC estimator using gaussian quadrature
    outputs:
        sigma: standard deviation of each fidelity per mode (nf,)
        rho: correlation coefficient per mode (nf,)
    '''

    nf = 2  # number of fidelities

    x, w = leggauss(npts) # draw GQ points and weights
    x = np.pi * x # rescale to [-pi, pi]
    w = np.pi * w

    # gnerate tensor grids for f(x1,x2,x3)
    X1, X2, X3 = np.meshgrid(x,x,x, indexing='ij') # (npts, npts, npts)
    X = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1) # (npts**3, 3)
    W = np.outer(np.outer(w,w),w).reshape(-1) # (npts**3,)

    y1 = f1(X).ravel() # (npts**3,)
    y2 = f2(X).ravel() # (npts**3,)

    mean1 = np.sum(W* y1) / (2*np.pi)**3 # note: denom is for pdf
    mean2 = np.sum(W* y2) / (2*np.pi)**3

    var1 = np.sum(W * (y1 - mean1)**2) / (2*np.pi)**3 # scalar
    var2 = np.sum(W * (y2 - mean2)**2) / (2*np.pi)**3 # scalar
    sigma = np.sqrt([var1, var2]) # (nf,)
    cov12 = np.sum(W * (y1 - mean1)*(y2 - mean2)) / (2*np.pi)**3 # scalar

    # get correlation coefficient between highfi and lowfis
    rho = np.zeros(nf+1) # (nf+1,)
    rho[0] = 1.0
    rho[1] = cov12 / (sigma[0]*sigma[1])
    print('sigma:', sigma, 'rho:', rho)
    np.savez("stats_ishi.npz", sigma=sigma, rho=rho)

def save_ishidata(fun, save_path, n_data=int(1e6), rid=42):
    '''
    This function saves Ishigami function data in h5 format.

    inputs:
    - fun: function handle
    - save_path: path to save the processed h5 file (string)
    - n_data: number of data to be generated
    '''

    rng = np.random.default_rng(rid)

    x = rng.uniform(-np.pi, np.pi, (int(n_data), 3)) # (n_data, d_in=3)
    y = fun(x).reshape(-1,1) # (n_data,1)

    # save necessary data
    with h5py.File(save_path, "w") as f:
        f.create_dataset("input", data=x, compression="gzip", compression_opts=9)
        f.create_dataset("output", data=y, compression="gzip", compression_opts=9)


def main():
    save_ishidata(f1, f'{data_dir}/highfi.h5', n_data=int(1e4))
    save_ishidata(f2, f'{data_dir}/lowfi.h5', n_data=int(1e4))
    save_ishidata(f1, f'{data_dir}/test.h5', n_data=int(1e3), rid=100)
    stats()

if __name__ == "__main__":
    main()


