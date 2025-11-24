# Multifidelity Kriging with budget allocation
This repository impelements hierarchical Kriging using two different allocation methods: 1) naive allocation and 2) multifidelity Monte Carlo (MFMC) allocation. The naive allocation allocates high and low fidelity samples using an arbitrary ratio $\tau$, where $\tau$ is a ratio of low fidelity samples to high fidelity samples. The goal of this project is to propose MFMC budget allocation strategy for Hierarchical Kriging to improve the prediction accuracy. We train and test Hierarchical Kriging model using MFMC allocations on: 1) Ishigami function example, 2) short column example, and 3) Finite element simulations of NASA CRM wing. The performance of MFMC allocation is compared with that of naive allocations with $\tau=2,4,8$.


## Prerequisites
1. Download the data for the third example.
* Download the wing structural stress data from [here](https://link.springer.com/article/10.1007/s00158-022-03274-1#Sec23) (3.9 GB) or by `wget https://static-content.springer.com/esm/art%3A10.1007%2Fs00158-022-03274-1/MediaObjects/158_2022_3274_MOESM1_ESM.zip`. Place the follwing data into `mfhikrig/data/wing`.
  1. `/data/crm_baseline_4DV_N1000_slim.h5`
  2. `/data/crm_coarse-grid_4DV_N1000_slim.h5`
  3. `/data/crm_coarse-ribs_4DV_N1000_slim.h5`


2. Preprocess the data
* For each example, run script beginning with `preproc_` to generate or format the data.


## Usage
Within each example folder, run `comparisonplot.py` to generate a relative error plot of HK with MFMC and naive allocations.
  * This script runs `errplot_mfmc` and `errplot_naive` in turn. Each of these script saves results in `npz` and these results will be loaded in `comparisonplot.py` to plot. If the `npz` files already exist, you can skip running these scripts by commenting out `import errplot_mfmc` and `import errplot_naive`
  * `errplot_mfmc`: trains HK model with MFMC allocation and computes the relative error. The results are saved in `npz`.
  * `errplot_naive`: trains HK model with naive allocations (using $\tau=2,4,8$) and computes the relative error. The results are saved in `npz`.
