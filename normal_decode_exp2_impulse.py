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

start_path = 'data/wolff/decodability_exp2_impulse'
save_path = 'data/wolff/decodability_exp2_impulse'

for i in range(1, 20):
    print("Doing " + str(i))
    
    file = 'dat_' + str(i) + '.mat'
    save_file_early1 = join(save_path, 'early1_' + str(i) + '.npy')
    save_file_early2 = join(save_path, 'early2_' + str(i) + '.npy')
    save_file_late1 = join(save_path, 'late1_' + str(i) + '.npy')
    save_file_late2 = join(save_path, 'late2_' + str(i) + '.npy')
    
    dat = scipy.io.loadmat(join(start_path, file))
    data_imp1_sess1 = dat['data_imp1_sess1']
    data_imp2_sess1 = dat['data_imp2_sess1']
    data_imp1_sess2 = dat['data_imp1_sess2']
    data_imp2_sess2 = dat['data_imp2_sess2']
    mem_angles_imp1_sess1 = dat['mem_angles_imp1_sess1'] # early at [:, 0], late at [:, 1]
    mem_angles_imp2_sess1 = dat['mem_angles_imp2_sess1']
    mem_angles_imp1_sess2 = dat['mem_angles_imp1_sess2']
    mem_angles_imp2_sess2 = dat['mem_angles_imp2_sess2']
    
    # Decoding session 1
    dec_imp1_early1, _ = wolff.similarity_p(data_imp1_sess1, mem_angles_imp1_sess1[:, 0], angspace, bin_width, 60)
    dec_imp1_late1, _ = wolff.similarity_p(data_imp1_sess1, mem_angles_imp1_sess1[:, 1], angspace, bin_width, 60)
    dec_imp2_early1, _ = wolff.similarity_p(data_imp2_sess1, mem_angles_imp2_sess1[:, 0], angspace, bin_width, 60)
    dec_imp2_late1, _ = wolff.similarity_p(data_imp2_sess1, mem_angles_imp2_sess1[:, 1], angspace, bin_width, 60)
    
    # Decoding session 2
    dec_imp1_early2, _ = wolff.similarity_p(data_imp1_sess2, mem_angles_imp1_sess2[:, 0], angspace, bin_width, 60)
    dec_imp1_late2, _ = wolff.similarity_p(data_imp1_sess2, mem_angles_imp1_sess2[:, 1], angspace, bin_width, 60)
    dec_imp2_early2, _ = wolff.similarity_p(data_imp2_sess2, mem_angles_imp2_sess2[:, 0], angspace, bin_width, 60)
    dec_imp2_late2, _ = wolff.similarity_p(data_imp2_sess2, mem_angles_imp2_sess2[:, 1], angspace, bin_width, 60)
    
    # Averaging impulse 1
    dec_imp1_early = (dec_imp1_early1.mean(0) + dec_imp1_early2.mean(0)) / 2
    dec_imp1_early = gaussian_filter(dec_imp1_early, sigma=8)
    dec_imp1_late = (dec_imp1_late1.mean(0) + dec_imp1_late2.mean(0)) / 2
    dec_imp1_late = gaussian_filter(dec_imp1_late, sigma=8)
    
    # Averaging impulse 2
    dec_imp2_early = (dec_imp2_early1.mean(0) + dec_imp2_early2.mean(0)) / 2
    dec_imp2_early = gaussian_filter(dec_imp2_early, sigma=8)
    dec_imp2_late = (dec_imp2_late1.mean(0) + dec_imp2_late2.mean(0)) / 2
    dec_imp2_late = gaussian_filter(dec_imp2_late, sigma=8)
    
    # Saving the results
    np.save(save_file_early1, dec_imp1_early)
    np.save(save_file_late1, dec_imp1_late)
    np.save(save_file_early2, dec_imp2_early)
    np.save(save_file_late2, dec_imp2_late)