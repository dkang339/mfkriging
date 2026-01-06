import numpy as np
from scipy.optimize import minimize
from .aux import KrigingAux

class DiscKriging(KrigingAux):

    def __init__(self, x_high, y_high, l_pred, 
                 kernel="ardmatern32", regularize=True, n_restart=1):
        super().__init__()
        self.d = x_high.shape[1]
        self.n = x_high.shape[0]
        self.kernel = kernel
        self.regularize = regularize
        self.n_restart = n_restart
        self.x_high = self._preproc(x_high)
        self.y_high = self._preproc(y_high)
        self.l_pred = self._preproc(l_pred)


    def neg_loglikeli(self, hyparas):
        '''
        objective function.
        evaluate negative log-likelihood for given hyperparameters.
        '''
        hyparas = np.asarray(hyparas)

        # transform back to original scale
        lscale = np.exp(hyparas[:self.d]) # length scale parameter

        if self.regularize:
            reg = np.exp(hyparas[self.d])**2 # regularaization parameter
        else:
            reg = len(self.x_high) * np.finfo(float).eps

        K = self._kernel_mat(self.x_high, self.x_high, lscale, kernel=self.kernel) # (n,n)
        K += reg * np.eye(len(K)) # add regularization (n,n)

        # cholesky decomposition
        L = np.linalg.cholesky(K) # (n,n)
        
        # ---------------------------------------------
        #     Solve generalized least squares (GLS)
        # ---------------------------------------------
        Linv_lpred = np.linalg.solve(L, self.l_pred)  # (n,1)
        Kinv_lpred = np.linalg.solve(L.T, Linv_lpred)  # (n,1)
        a = self.l_pred.T @ Kinv_lpred   # (1,1)

        Linv_yhigh = np.linalg.solve(L, self.y_high)  # (n,1)
        Kinv_yhigh = np.linalg.solve(L.T, Linv_yhigh) # (n,1)
        b = self.l_pred.T @ Kinv_yhigh   # (1,1)

        # compute rho
        rho = np.linalg.solve(a, b) # (1,1)
        
        # ---------------------------------------------
        #     Compute negative log-likelihood
        # ---------------------------------------------
        y_disc = self.y_high - self.l_pred @ rho # (n,1)

        # compute log-likelihood: y^T K^-1 y
        a = np.linalg.solve(L, y_disc) # (n,1)
        b = np.linalg.solve(L.T, a) # (n,1)
        elem1 = np.dot(y_disc.T, b) # (1,1)

        # compute process variance
        var = elem1 / self.n # (1,)

        # compute log-likelihood: log|K|
        diags = np.diag(L) # (n,)
        elem2 = 2 * np.sum(np.log(diags)) # (1,)

        loglikeli = 0.5 * (self.n* np.log(var) +  elem2)

        return loglikeli.item()
    

    def train(self):

        # ---------------------------------------------
        #           optimize hyperparameters
        # ---------------------------------------------
        # bound constraints
        bounds = self.get_bounds(self.x_high, self.d)
        lb = bounds[:,0]
        ub = bounds[:,1]

        # initial hyperparameters
        hyparas0 = self.init_hyparas(bounds, self.d)

        best_loglikeli = np.inf
        for _ in range(self.n_restart): # repeat optimization n_restart times
            hyparas0 = np.random.uniform(lb, ub) # random initial guess
            res = minimize(self.neg_loglikeli, hyparas0, method='SLSQP', bounds=bounds)
            if res.fun < best_loglikeli: # save the best hyperparameters
                best_loglikeli = res.fun
                best_hyparas = res.x

        # save hyperparameters
        self.lscale = np.exp(best_hyparas[:self.d]) # length scale parameter
        if self.regularize:
            self.reg = np.exp(best_hyparas[self.d])**2 # regularaization parameter
        else:
            self.reg = len(self.x_high) * np.finfo(float).eps

        # ---------------------------------------------
        #       save L, weights, rho for prediction
        # ---------------------------------------------
        # compute kernel matrix btw training points
        K = self._kernel_mat(self.x_high, self.x_high, self.lscale, 
                       kernel=self.kernel) # (n,n)
        K += self.reg * np.eye(len(K)) # add regularization (n,n)

        # cholesky decomposition
        self.L = np.linalg.cholesky(K) # (n,n)

        # compute rho
        Linv_lpred = np.linalg.solve(self.L, self.l_pred)  # (n,1)
        Kinv_lpred = np.linalg.solve(self.L.T, Linv_lpred)  # (n,1)
        a = self.l_pred.T @ Kinv_lpred   # (1,1)
        Linv_yhigh = np.linalg.solve(self.L, self.y_high)  # (n,1)
        Kinv_yhigh = np.linalg.solve(self.L.T, Linv_yhigh) # (n,1)
        b = self.l_pred.T @ Kinv_yhigh   # (1,1)
        self.rho = np.linalg.solve(a, b) # (1,1)

        # compute weights
        self.y_disc = self.y_high - self.l_pred @ self.rho # (n,1)
        Linv_ydisc = np.linalg.solve(self.L, self.y_disc) # (n,1)
        self.weight = np.linalg.solve(self.L.T, Linv_ydisc) # = (K + reg*I)^-1 * y, (n,1) 


    def predict(self, x_test, yl_test, return_var=False):
        '''
        predict at unseen points.
        compute posterior mean and variance.
        '''
        # preprocess data
        x_test = self._preproc(x_test)

        # compute kernel matrix btw test and training points
        k = self._kernel_mat(self.x_high, x_test, self.lscale, kernel=self.kernel) # (n,m)

        # compute posterior mean
        disc_pred = k.T @ self.weight # (m,1)
        y_pred = yl_test @ self.rho + disc_pred # (m,1)

        if return_var:
            # compute posterior variance
            v = np.linalg.solve(self.L, k) # (n,m)
            elem1 = self._kernel_mat(x_test, x_test, self.lscale, kernel=self.kernel).diagonal() - np.sum(v**2, axis=0) # (m,)
            
            Linv_F = np.linalg.solve(self.L, self.l_pred) # (n,1)
            Kinv_F = np.linalg.solve(self.L.T, Linv_F) # (n,1)
            F_Kinv_F = self.l_pred.T @ Kinv_F # (1,1)
            
            k_Kinv_F = k.T @ Kinv_F # (m,1)
            diff = k_Kinv_F - yl_test # (m,1)
            
            elem2 = np.linalg.solve(F_Kinv_F,diff.T) # (1,m)
            elem2 = (diff @ elem2).diagonal() # (m,)

            var_pred = (elem1 + elem2).reshape(-1, 1) # (m,1)
            
            return y_pred, var_pred
        else:
            return y_pred
        