import numpy as np
from .kriging import Kriging
from .disc_kriging import DiscKriging
from .aux import KrigingAux

class MFKriging(KrigingAux):
    '''
    Train low fidelity and discrepancy GPs and 
    assemble them using Kennedy O'Hagan approach.
    '''
    def __init__(self, kernel="ardmatern32", regularize=True, normalize=True, n_restart=10):
        self.kernel = kernel
        self.regularize = regularize
        self.normalize = normalize
        self.n_restart = n_restart
        self.lowfi = None
        self.discrepancy = None
        self.rho = None

    def train(self, x_low, y_low, x_high, y_high):
        '''
        train hierarchical Kriging model.
        
        inputs
        - x_low: low-fidelity input data, (m, d)
        - y_low: low-fidelity output data, (m, 1)
        - x_high: high-fidelity input data, (n, d)
        - y_high: high-fidelity output data, (n, 1)
        '''

        # preprocess data
        x_low, y_low = self._preproc(x_low), self._preproc(y_low)
        x_high, y_high = self._preproc(x_high), self._preproc(y_high)
        
        if self.normalize:
            x_high, self.x_mean, self.x_std = self._normalize_data(x_high)
            x_low = (x_low - self.x_mean) / self.x_std
            y_high, self.y_mean, self.y_std = self._normalize_data(y_high)
            y_low, _, _ = self._normalize_data(y_low)
        else:
            self.x_mean, self.x_std = None, None
            self.y_mean, self.y_std = None, None

        # ---------------------------------------------
        #           train low-fidelity GP
        # ---------------------------------------------
        self.lowfi = Kriging(kernel=self.kernel, 
                             regularize=self.regularize, normalize=False, 
                             n_restart=self.n_restart)
        self.lowfi.train(x_low, y_low)

        # ---------------------------------------------
        #          get low fidelity prediction
        # ---------------------------------------------
        l_pred = self.lowfi.predict(x_high) # (n,1)


        # ---------------------------------------------
        #           train discrepancy GP
        # ---------------------------------------------
        self.discrepancy = DiscKriging(x_high, y_high, l_pred,
                                       kernel=self.kernel,
                                        regularize=self.regularize,
                                        n_restart=self.n_restart)
        self.discrepancy.train()

    
    def predict(self, x_test, return_var=False):
        '''
        predict at unseen points.
        compute posterior mean and variance.

        inputs
        - x_test: test input data, (n_test, d)
        - return_var: whether to return posterior variance
        '''

        # preprocess data
        x_test = self._preproc(x_test)
        if self.normalize:
            x_test = (x_test - self.x_mean) / self.x_std

        y_low = self.lowfi.predict(x_test) # (n_test, 1)

        if return_var:
            y_hf, var_hf = self.discrepancy.predict(x_test, y_low, return_var=True)
        else:
            y_hf = self.discrepancy.predict(x_test, y_low)
            var_hf = None

        if self.normalize:
            y_hf = y_hf * self.y_std + self.y_mean # unnormalize prediction
            if var_hf is not None:
                var_hf = var_hf * self.y_std**2 # unnormalize prediction variance
                return y_hf, var_hf
            else:
                return y_hf
        else:
            if var_hf is not None:
                return y_hf, var_hf
            else:
                return y_hf