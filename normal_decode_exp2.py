import numpy as np
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output, display
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from sklearn.cluster import KMeans
import multiprocessing
from functools import partial
import time
import os
from os.path import join
import re

import wolff
import wolff_cross

bin_width = np.pi / 6
angspace = np.arange(-np.pi, np.pi, bin_width)

start_path = '/Users/s3344282/Analysis/full_model/exp2_original_AS_model'

for _folder in range(1, 20):
    folder = str(_folder)
    print("Doing " + folder)
    
    path = join(start_path, folder)
    file1 = join(path, "cos_amp1.npy")
    file2 = join(path, "cos_amp2.npy")
    
    angles = np.load(join(path, "angles.npy"))
    
    if not os.path.exists(file1):
        print("Decoding data 1")
        data = np.load(join(path, "data1.npy"))

        cos_amp, _ = wolff.similarity_p(data, angles, angspace, bin_width, 40)

        cos_amp = np.mean(cos_amp, 0)
        np.save(file1, cos_amp)
        
    if not os.path.exists(file2):
        print("Decoding data 2")
        data = np.load(join(path, "data2.npy"))

        cos_amp, _ = wolff.similarity_p(data, angles, angspace, bin_width, 40)

        cos_amp = np.mean(cos_amp, 0)
        np.save(file2, cos_amp)
        
    print("Done with " + folder)