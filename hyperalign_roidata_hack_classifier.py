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

#specify the start and end points of each event we want to classify
startpoints=[91,507,681]
endpoints=[111,526,698]

numTRs=1976
numevents=len(startpoints)

#make a list of events per TR
scenes_to_classify=np.zeros(numTRs)
for i in range(0,numevents,1):
   for j in range(startpoints[i],endpoints[i]+1,1): 
       scenes_to_classify[j]=i+1

print('scenes')
fig, ax = plt.subplots()
ax.plot(range(1,numTRs+1,1), scenes_to_classify)

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
    ds.sa["targets"]=scenes_to_classify
    run_datasets.append(ds)

#run hyperalignment
hyper = Hyperalignment()
hypmaps = hyper(run_datasets)

#apply it to the data
ds_hyper = [h.forward(sd) for h, sd in zip(hypmaps, run_datasets)]

#get data properties
grouplist=range(0,len(ds_hyper),1)
numvox=ds_hyper[1].shape[1]
numsubjects=len(ds_hyper)

#make a 3d data matrix, subject, TR, voxel
data3d=np.zeros((numsubjects,numTRs,numvox))
for i in range(0,numsubjects,1):
    data3d[i,:,:]=np.array(ds_hyper[i])
    
accuracy=np.zeros((numsubjects,numevents))
for i in range(0,numsubjects,1):
    
    #get the matrices for the specifc subject and the subgroup
    group=np.setdiff1d(grouplist,i)
    datagroup=data3d[group,:,:]
    datasub=data3d[i,:,:]
    
    #intitialize event patterns
    eventpatterns_group=np.zeros((numvox,numevents))
    eventpatterns_sub=np.zeros((numvox,numevents))
    
    # get average event patterns for subject and subgroup
    for e in range(1,numevents):
        eventindex=np.nonzero(scenes_to_classify==e)
        eventpatterns_group[:,e]=np.mean(datagroup[:,eventindex[0],:],(0,1))
        eventpatterns_sub[:,e]=np.mean(datasub[eventindex[0],:],(0))
        
    # loop over events of the subject    
    for e in range(1,numevents):   
        correl=np.zeros(numevents)
        #loop over events of the subgroup and correlate with subject
        for e2 in range(1,numevents): 
            c=np.corrcoef(eventpatterns_sub[:,e],eventpatterns_group[:,e2])
            correl[e2]=c[0,1]
            
        # find highest correlation
        label=np.argmax(correl)
        
        #check accuracy
        if label==e:
            accuracy[i,e]=1

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