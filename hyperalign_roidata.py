import os
import scipy.io
import numpy as np
import re
from mvpa2.suite import *
import matplotlib.pyplot as plt
from scipy import stats

folder = "C:/Users/Linda/MIND_python/mind_2018-master/tutorials/sherlock_nifti_kit_v2_withdata/subjects/"
subfolders = [f.path for f in os.scandir(folder) if f.is_dir() ]   

roifile='pmc_nn'

#load the data for all subjects for one ROI file
#the data has 1976 timepoints and 481 voxels
run_datasets=[];
for i in subfolders:
    #load data 
    res = re.findall("subjects/s([0-9]+)", i)
    mat=scipy.io.loadmat(i + '/sherlock_movie/' + roifile + '_sherlock_movie_s' + res[0])
    data=np.transpose(mat['rdata'])
    
    # put into pymvpa structure
    ds=dataset_wizard(data)
    run_datasets.append(ds)

#run hyperalignment
hyper = Hyperalignment()
hypmaps = hyper(run_datasets)

#apply it to the data
ds_hyper = [h.forward(sd) for h, sd in zip(hypmaps, run_datasets)]

# visualize some changes
sub1_orig=np.array(run_datasets[1])
sub1_hyper=np.array(ds_hyper[1])

# time*time similarity matrix - which should not change with hyperalignment
time_cmat_orig=np.corrcoef(sub1_orig)
time_cmat_hyper=np.corrcoef(sub1_hyper)

# vox*vox similarity matrix - which should change with hyperalignment
vox_cmat_orig=np.corrcoef(sub1_orig.T)
vox_cmat_hyper=np.corrcoef(sub1_hyper.T)

print('original data')
fig, ax = plt.subplots()
ax.imshow(sub1_orig.T, cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=1)

print('hyperaligned data')
fig, ax = plt.subplots()
ax.imshow(sub1_hyper.T, cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=1)

print('original data time correlation matrix')
fig, ax = plt.subplots()
ax.imshow(time_cmat_orig, cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=0.5)

print('hyperaligned data time correlation matrix')
fig, ax = plt.subplots()
ax.imshow(time_cmat_hyper, cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=0.5)

print('original data voxel correlation matrix')
fig, ax = plt.subplots()
ax.imshow(vox_cmat_orig, cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=1)

print('hyperaligned data voxel correlation matrix')
fig, ax = plt.subplots()
ax.imshow(vox_cmat_hyper, cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=1)

print('differences in voxel correlation matrix')
fig, ax = plt.subplots()
cax=ax.imshow(vox_cmat_orig-vox_cmat_hyper, cmap=plt.get_cmap('coolwarm'), interpolation='nearest',
               vmin=-0.5, vmax=0.5)
cbar = fig.colorbar(cax, ticks=[-0.5, 0, 0.5])
cbar.ax.set_yticklabels(['< -0.5', '0', '> 0.5'])  # vertically oriented colorbar

print('differences in time correlation matrix')
fig, ax = plt.subplots()
cax=ax.imshow(time_cmat_orig-time_cmat_hyper, cmap=plt.get_cmap('coolwarm'), interpolation='nearest',
               vmin=-0.2, vmax=0.2)
cbar = fig.colorbar(cax, ticks=[-0.3, 0, 0.3])
cbar.ax.set_yticklabels(['< -0.3', '0', '> 0.3'])  # vertically oriented colorbar