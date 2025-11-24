'''
Preprocess wing data.
'''

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
import numpy as np
import h5py
import sys
from pathlib import Path
import os
current_dir = Path(__file__).parent # get current directory
root_dir = current_dir.parent.resolve() # get src directory
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / '../src'))
data_dir = '../../../data/wing'
os.makedirs(data_dir, exist_ok=True)
from mfmc import MFMC

fname_h = '../../../data/wing/crm_baseline_4DV_N1000_slim.h5'
fname_l = '../../../data/wing/crm_coarse-ribs_4DV_N1000_slim.h5'

nf = 2  # number of fidelities
w = np.array([1, 0.8704]).T # cost per each fidelity (nf,)

def primary_wing(fpath,save_path=None):
    '''
    The original wing dataset includes the data for primary wing and tail wing.
    This function extracts the primary wing elements only based on element connectivity.
    '''

    # load Data
    with h5py.File(fpath, "r") as f:
        points_all = f["GRID/Points"][:]
        points = points_all[0] # get nodes of the first sample
        cells = f["GRID/Cells"][:].reshape(-1, 4)  # connectivity (num_elem, 4)
        von_mises = f["GRID/CELL_DATA/vonMisesStress_2p5g_psi"][:]  # stress (1000, num_elem)
    
    # get element connectivity (skip the first column - its element id))
    elements = cells[:, 1:4]  # (num_cells, 3)
    num_elements = elements.shape[0]
    
    # build an element connectivity graph
    # two elements are connected if they share an edge (i.e. share at least 2 nodes).
    edge_to_elements = defaultdict(list)
    for elem_idx, elem in enumerate(elements):
        sorted_elem = np.sort(elem)
        # define the three edges for a triangular element
        edges = [
            (sorted_elem[0], sorted_elem[1]),
            (sorted_elem[0], sorted_elem[2]),
            (sorted_elem[1], sorted_elem[2])
        ]
        for edge in edges:
            edge_to_elements[edge].append(elem_idx)
    
    rows = []
    cols = []
    for edge, elems in edge_to_elements.items():
        if len(elems) > 1:
            # connect every pair of elements that share this edge
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    rows.extend([elems[i], elems[j]])
                    cols.extend([elems[j], elems[i]])
    
    # construct the sparse matrix for element connectivity
    element_graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_elements, num_elements))
    
    # find connected components on the element graph
    _, comp_labels = connected_components(element_graph)
    
    # identify primary wing that has max num of elements
    comp_idx, counts = np.unique(comp_labels, return_counts=True)
    pwing_idx = comp_idx[np.argmax(counts)]
    
    # filter elements (and stress) that belong to primary wing
    element_mask = comp_labels == pwing_idx
    filtered_elements = elements[element_mask]
    filtered_stress = von_mises[:, element_mask]
    unique_nodes = np.unique(filtered_elements)
    
    # remap old node indices to a new contiguous range
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_nodes)}
    remapped_elements = np.vectorize(node_mapping.get)(filtered_elements)
    
    # filter the points array to include only nodes used in the primary wing
    filtered_points = points[unique_nodes]
    
    # save the mesh file of filtered component
    if save_path: 
        with h5py.File(save_path, "w") as f:
            f.create_dataset("GRID/Points",data=filtered_points)
            f.create_dataset("GRID/Cells",data=remapped_elements)


    return filtered_points, remapped_elements, filtered_stress

def save_wingdata(fname, save_path, ifscalar=False):
    '''
    This function saves necessary data of primary wing only in h5 format.

    inputs:
    - fname: original h5 file path (string)
        - e.g., '../../data/wing/crm_baseline_4DV_N1000_slim.h5'
    - save_path: path to save the processed h5 file (string)
        - e.g., '../../data/wing/highfi.h5'
    '''

    # read primary wing data - nodes, elements, von Mises stress
    node, elem, vm = primary_wing(fname) # vm: (n_data, d)

    if ifscalar: # convert to scalar output
        vm = vm.max(axis=1)
    else:
        pass
    
    # read inputs
    with h5py.File(fname, "r") as f:
        input = f["STATE/DV_Values"][:]  # (n_data, d_in)

    # save necessary data
    with h5py.File(save_path, "w") as f:
        f.create_dataset("input", data=input, compression="gzip", compression_opts=9)
        f.create_dataset("node", data=node, compression="gzip", compression_opts=9)
        f.create_dataset("elem", data=elem, compression="gzip", compression_opts=9)
        f.create_dataset("output", data=vm, compression="gzip", compression_opts=9)
        

# save datasets
save_wingdata(fname_h, f'{data_dir}/highfi.h5',ifscalar=True)
save_wingdata(fname_l, f'{data_dir}/lowfi_rib.h5',ifscalar=True)
mfmc = MFMC(f'{data_dir}/highfi.h5',f'{data_dir}/lowfi_rib.h5',w)
sigma, rho = mfmc.stats()
print('std dev:', sigma)
print('correlation coeff:', rho)
np.savez("stats_wing.npz", sigma=sigma, rho=rho)