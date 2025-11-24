'''
This script loads the saved error data and makes a plot.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "font.size": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14
        })
else:
    plt.rcdefaults()    

model_stats = np.load('stats_wing.npz', allow_pickle=True)
sigma, rho = model_stats["sigma"], model_stats["rho"]
print('sigma:', sigma, 'rho:', rho)

# import errplot_mfmc
# import errplot_naive

# TODO: specify the output file names
model_orig = np.load('npz/err_naive_tau.npz', allow_pickle=True)
model_mfmc = np.load('npz/err_mfmc.npz', allow_pickle=True)

# common parameters
p = model_mfmc['model'].item()['p']
p = p * 5.4 # scale back to cpu seconds
print('p:', p)

tau = model_orig['model'].item()['tau']
n_p, n_tau = len(p), len(tau)
w = model_orig['model'].item()['w']
# r = model_orig['model'].item()['r']

# original allocation
p_orig = model_orig['model'].item()['p'] # (n_p,)
p_orig = np.array(p_orig) * 5.4 # scale back to cpu seconds
n_orig = model_orig['model'].item()['n_all'] # (n_p, n_tau)
m_orig = model_orig['model'].item()['m_all'] # (n_p, n_tau)
err_orig = model_orig['err_orig'] # (n_p, rep, n_tau)
meanerr_orig = model_orig['mean_err'] # (n_p, n_tau)
sigma_orig = np.std(err_orig, axis=1) # (n_p, n_tau)

# modified allocation
n_mfmc = model_mfmc['model'].item()['n'] # (n_p,)
m_mfmc = model_mfmc['model'].item()['m_mf'] # (n_p,)
err_mfmc = model_mfmc['err_mfmc'] # (n_p, rep)
meanerr_mfmc = model_mfmc['mean_err_mfmc'] # (n_p,)
sigma_mfmc = np.std(err_mfmc, axis=1)
ratio = model_mfmc['model'].item()['ratio'] #(nf,)
ratio = ratio[1] # ratio of lowfi to highfi samples

print('meanerr_orig:', meanerr_orig)
print('meanerr_mfmc:', meanerr_mfmc)
print('n_orig:', n_orig, 'm_orig:', m_orig)
print('n_mfmc:', n_mfmc, 'm_mfmc:', m_mfmc)
print('ratio:', ratio)


ax = plt.gca()
for i, tau in enumerate(tau):
    plt_orig, = ax.plot(p_orig, meanerr_orig[:,i], ':o', label=rf'Naive allocation ($\tau = {tau}$)')
plt_mfmc, = ax.plot(p, meanerr_mfmc, '-o', label=rf'MFMC allocation ($\tau={ratio:.3f}$)')
ax.legend()
plt.xlabel('Computational budget (CPU s)')
plt.ylabel('Mean relative error')
plt.savefig('plots/err_comparison.png',dpi=600,bbox_inches='tight') # save figure