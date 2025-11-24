import numpy as np
import sys
from pathlib import Path
import h5py

current_dir = Path(__file__).parent # get current directory
root_dir = current_dir.parent.resolve() # get src directory
sys.path.append(str(root_dir / 'src'))


class MFMC:
    '''
    Multifidelity Monte Carlo allocation
    '''
    def __init__(self,fname_h,fname_l,cost):
        '''
        inputs:
            cost: cost per each fidelity model (nf,)
            fname_h: file name containing high fidelity data
            fname_l: file name containing low fidelity data
        '''

        self.nf = 2  # number of fidelities
        self.w = cost # cost per each fidelity (nf,)
    
        # load data
        with h5py.File(fname_h, "r") as f:
            h_out = f["output"][:]
        self.h_out = h_out.T # (d,N), d: number of high fidelity spatial dimensions
        with h5py.File(fname_l, "r") as f:
            l_out = f["output"][:]
        self.l_out = l_out.T # (q,N), q: number of low fidelity spatial dimensions
        
        if h_out.ndim == 1 and l_out.ndim == 1:
            self.h_out = h_out.reshape(1, -1)
            self.l_out = l_out.reshape(1, -1)

    def stats(self):
        ''''
        compute second-order stats
        outputs:
            sigma: standard deviation of each fidelity per fidelity (nf,)
            rho: correlation coefficient per fidelity (nf,)
        '''
        # get standard deviation of each fidelity
        sigma = np.zeros(self.nf) # (nf,)
        sigma[0] = np.std(self.h_out, axis=1, ddof=1) # highfi (scalar)
        sigma[1] = np.std(self.l_out, axis=1, ddof=1) # lowfi (scalar)

        # get correlation coefficient between highfi and lowfis
        rho = np.zeros(self.nf+1) # (nf+1,)
        rho[0] = np.corrcoef(self.h_out.squeeze(), self.h_out.squeeze())[0,1]
        rho[1] = np.corrcoef(self.h_out.squeeze(), self.l_out.squeeze())[0,1]

        return sigma, rho


    def alloc(self, sigma, rho, p):
        '''
        allocate samples across different fidelities
        inputs:
            sigma: standard deviation of each fidelity per fidelity (nf,)
            rho: correlation coefficient per fidelity (nf,)
        outputs:
            m: number of samples for each fidelity (nf,)
        '''

        m = np.zeros(self.nf)
        temp = rho[:-1]*sigma[0]/sigma[:]
        temp = (rho[:-1]**2 - rho[1:]**2)
        const = np.sqrt(self.w[0]*temp/(self.w*(1-rho[1]**2))) # (nf,)
        m[0] = p/(self.w.transpose() @ const)
        m[1] = m[0]*const[1:]
        m = np.floor(m)
        m = m.astype(int)

        return m, const