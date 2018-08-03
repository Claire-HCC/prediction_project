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
run_datasets=[];
for i in subfolders:
    res = re.findall("subjects/s([0-9]+)", i)
    mat=scipy.io.loadmat(i + '/sherlock_movie/' + roifile + '_sherlock_movie_s' + res[0])
   # data=stats.zscore(mat['rdata'], axis=0)
    data=mat['rdata']
    ds=dataset_wizard(data)
    #fig, ax = plt.subplots()
    #ax.imshow(data, cmap=plt.get_cmap('hot'), interpolation='nearest',
    #           vmin=0, vmax=1)
    run_datasets.append(ds)

#run hyperalignment
hyper = Hyperalignment()
hypmaps = hyper(run_datasets)

#apply it to the data
ds_hyper = [h.forward(sd) for h, sd in zip(hypmaps, run_datasets)]



