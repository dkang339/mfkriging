import numpy as np
from .kernels import KERNELS
from scipy.spatial.distance import pdist

class KrigingAux:
    '''
    Defines auxiliary functions for Kriging.
    '''

    def _preproc(self, x):
        # ensure data in 2D array
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        assert x.ndim == 2
        return x


    def _kernel_mat(self, Z1, Z2, lscale, kernel):
        # compute kernel matrix
        return KERNELS[kernel](Z1, Z2, lscale)


    def _normalize_data(self, x):

        x = np.asarray(x)
        if x.ndim == 1: # ensure data in 2D array
            x = x.reshape(-1, 1)
        # get mean and std of data
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)

        # ensure scalar stats for 1D input
        if x.shape[1] == 1:
            x_mean = x_mean.item()
            x_std = x_std.item()

        # normalize data
        x_norm = (x - x_mean) / x_std

        return x_norm, x_mean, x_std
    

    def init_hyparas(self, bounds, d, y=None):
        '''
        Initialize hyperparameters.
        Adapted from pyToolBox's hyperparameter initialization
        (downloadable from https://link.springer.com/article/10.1007/s00158-022-03274-1#Sec23).

        inputs
            - bounds: (lower bound, upper bound) in log space for hyperparameters
            - d: input dimension
            - y: training output data (n, 1) for data-dependent noise initialization
        '''

        bounds = np.asarray(bounds)
        hyparas0 = np.mean(bounds, axis=1) # (d+1,)

        # data-dependent noise initialization (matching MAROM-MF-Structure)
        if y is not None:
            tiny = np.finfo(float).tiny
            hyparas0[d] = np.log(0.1 * np.var(y) + tiny)

        return hyparas0

    def get_bounds(self, x, d, y=None):
        '''
        Set bound constraints for hyperparameters.
        Adapted from pyToolBox's hyperparameter initialization
        (downloadable from https://link.springer.com/article/10.1007/s00158-022-03274-1#Sec23).

        inputs
            - x: training data (n, d)
            - d: input dimension
            - y: training output data (n, 1) for data-dependent noise bounds
        outputs
            - bounds: log-scaled (lower bound, upper bound)
        '''
        x = np.asarray(x)

        r = pdist(x, 'euclidean')
        r = r[r > 0] # avoid zero distances
        rmin = np.min(r)
        rmax = np.max(r)

        # set lengthscale bounds
        lb_lscale = np.log(rmin / 10)
        ub_lscale = np.log(10 * rmax)
        length_bounds = np.vstack([ [lb_lscale, ub_lscale] for _ in range(d) ])

        # set noise bounds (data-dependent, matching MAROM-MF-Structure)
        npts = x.shape[0]
        cond = npts * np.finfo(float).eps
        if y is not None:
            y = np.asarray(y)
            reg_lb = np.log(cond)
            reg_ub = np.log(np.ptp(y)**2 + cond)
        else:
            reg_lb = np.log(1e-5)
            reg_ub = np.log(1e-1)
        reg_bounds = np.array([[reg_lb, reg_ub]])

        # combine bounds
        bounds = np.vstack([length_bounds, reg_bounds])

        return bounds
