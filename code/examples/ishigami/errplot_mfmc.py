'''
Run ishigami function example for MFMC allocation 
and rho chosen from kriging.
'''

import subprocess
import numpy as np
import time
import os
import sys
from pathlib import Path
current_dir = Path(__file__).parent # get current directory
root_dir = current_dir.parent.resolve() # get src directory
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / '../src'))
from mfmc import MFMC
from hikrig import hikrig
plt_dir = f'plots'
npz_dir = f'npz'
os.makedirs(plt_dir, exist_ok=True)
os.makedirs(npz_dir, exist_ok=True)
import preproc_ishi


# ---------------------------------------------
#                   set model
# ---------------------------------------------
subprocess.run(["python3", "preproc_ishi.py"]) # run preproc_ishi to generate data

# TODO: specify which lowfi data to use
fname_h = '../../data/ishi/highfi.h5'
fname_l = '../../data/ishi/lowfi.h5'
fname_test = '../../data/ishi/test.h5'
f_data = [fname_h, fname_l]

w = np.array([1, 0.1]).T # cost per each fidelity (nf,)
rep = 100 # number of sample replicates
nf = 2 # number of fidelities
p = np.array([10, 20, 30, 50, 100, 200, 300]) # total budget
model = {
    "fname_h": fname_h,
    "fname_l": fname_l,
    "fname_test": fname_test,
    "w": w,
    "nf": nf,
    "rep": rep,
    "p": p
}
print('budget:', p)


# ---------------------------------------------
#                run simulation
# ---------------------------------------------
# load mfmc class for budget allocation
mfmc = MFMC(fname_h,fname_l,w)

# compute m using mfmc
dat = np.load("stats_ishi.npz", allow_pickle=True)
sigma, rho = dat['sigma'], dat['rho']

m_mf = np.zeros((nf,len(p)),dtype=int) # number of samples for each fidelity by mfmc
for i in range(len(p)):
    m_mf[:,i], ratio = mfmc.alloc(sigma, rho, p[i])
model["n"] = m_mf[0,:]
model["m_mf"] = m_mf[1,:]
model["ratio"] = ratio
print('# of highfi samples:', model["n"])
print('corresponding # of lowfi samples:', model["m_mf"])
print('ratio of lowfi to highfi samples:', ratio)

# run hikrig with mfmc m
err_mfmc = np.zeros((len(p),rep))
stime = time.time()
for k in range(rep):
    print('rep: ', k)
    for i in range(len(p)):
        # --- hikrig with MFMC m ---
        err_mfmc[i,k] = hikrig(model,m_mf[0,i],m_mf[1,i],k)
etime = time.time()
print('time:', etime-stime)

# compute mean error and its std
meanerr_mfmc = np.mean(err_mfmc, axis=1)
print('mean_err_mfhikrig:', meanerr_mfmc)
sigma_mfmc = np.std(err_mfmc, axis=1)
np.savez(f'{npz_dir}/err_mfmc.npz',    # save error data
         err_mfmc=err_mfmc, mean_err_mfmc=meanerr_mfmc,
         model=model)
