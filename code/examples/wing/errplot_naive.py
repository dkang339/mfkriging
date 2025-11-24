'''
Run hierarchical Kriging for structural stress wing example (single output case) 
for different tau values.
'''

import numpy as np
import time
import os
import sys
from pathlib import Path
current_dir = Path(__file__).parent # get current directory
root_dir = current_dir.parent.resolve() # get src directory
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / '../src'))
from hikrig import hikrig
plt_dir = f'plots'
npz_dir = f'npz'
os.makedirs(plt_dir, exist_ok=True)
os.makedirs(npz_dir, exist_ok=True)


# ---------------------------------------------
#                   set model
# ---------------------------------------------
# TODO: specify which lowfi data to use
fname_h = '../../../data/wing/highfi_so.h5'
fname_l = '../../../data/wing/lowfi_rib_so.h5'
w = np.array([1, 0.8704]).T # cost per each fidelity (nf,)
rep = 100 # number of sample replicates
nf = 2
tau = np.array([2, 4, 8]) # ratio of lowfi to highfi samples
p = np.array([200, 300, 400, 500, 600, 700, 800, 850]) # total budget

model = {
    "fname_h": fname_h,
    "fname_l": fname_l,
    "w": w,
    "nf": nf,
    "rep": rep,
    "tau": tau,
    "p": p
}

n_all = np.zeros((len(p), len(tau))) # number of highfi samples
m_all = np.zeros((len(p), len(tau))) # number of lowfi samples
for j in range(len(tau)):
    n_all[:,j] = p/(w[0] + w[1]*tau[j])
    m_all[:,j] = tau[j]*n_all[:,j] # number of lowfi samples
n_all = np.floor(n_all).astype(int)
m_all = np.floor(m_all).astype(int)
model["n_all"] = n_all
model["m_all"] = m_all
print('n_all:', n_all)
print('m_all:', m_all)


# ---------------------------------------------
#                run simulation
# ---------------------------------------------
err = np.zeros((len(p),rep,len(tau)))
stime = time.time()
for j in range(len(tau)):
    n = n_all[:,j]
    m = m_all[:,j] # number of lowfi samples for original marom
    for k in range(rep):
        print('rep: ', k)
        for i in range(len(n)):
            # --- original hikrig ---
            err[i,k,j] = hikrig(model,n[i],m[i],k,split=True)
etime = time.time()
print('time:', etime-stime)

# compute mean error
mean_err = np.mean(err, axis=1)
print('mean_err_hikrig:', mean_err)
sigma = np.std(err, axis=1)

np.savez(f'{npz_dir}/err_orig_tau_p_so.npz',    # save error data
         err_orig=err, mean_err=mean_err,
         model=model)

