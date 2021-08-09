import time
import watchdog
from watchdog.observers import Observer
import numpy as np
from sklearn.cluster import KMeans
import re
import os
from os.path import join
from enum import Enum, auto

import wolff_cross

neuro_pat1 = re.compile(r"neuron_data_first")
neuro_pat2 = re.compile(r"neuron_data_second")
angle_pat = re.compile(r"initial_angles")
available_neuro1 = {}
available_neuro2 = {}
available_angle  = {}

path = '/Users/s3344282/Decoding/full_model/exp2_original_AS_model'
save_path = '/Users/s3344282/Analysis/full_model/exp2_original_AS_model'

NUM_PARTS = 6

class FileType(Enum):
    DATA   = auto()
    ANGLES = auto()
    
# Group the thousands of neurons into a handful of channels 
def group(mem_data):
    cut_data = mem_data[:, :250, :] # trials by 500 by neurons
    num_channels = 17
    neurons = np.mean(cut_data, 1).T # neurons by trials
    kmeans = KMeans(n_clusters=num_channels, n_init=20, tol=1e-20).fit(neurons)
    
    data = np.empty((mem_data.shape[0], num_channels, mem_data.shape[1])) # trials by num_channels by timesteps
    for channel in range(num_channels):
        print(str(channel + 1) + "/" + str(num_channels), end='\r')
        data[:, channel, :] = np.mean(mem_data[:, :, kmeans.labels_ == channel], axis=2)
    
    return data

def process_files(available_dic, sub):
    time.sleep(1)
    print("Processing some data of subject " + sub)
    
    save_dir = join(save_path, sub)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if available_dic == available_neuro1:
        ft = FileType.DATA
        save_file = join(save_dir, "data1.npy")
        save_sigma = join(save_dir, "sigma1.npy")
    elif available_dic == available_neuro2:
        ft = FileType.DATA
        save_file = join(save_dir, "data2.npy")
        save_sigma = join(save_dir, "sigma2.npy")
    else:
        ft = FileType.ANGLES
        save_file = join(save_dir, "angles.npy")
        
    parts = sorted(list(available_dic[sub]))
    part = load_file(parts[0])
    raw_data = np.empty((NUM_PARTS,) + part.shape, dtype=part.dtype)
    raw_data[0] = part
        
    for i, file in enumerate(parts[1:]):
        raw_data[i+1] = load_file(file)
        
    shape = list(part.shape)
    shape.pop(0)
    raw_data = raw_data.reshape([-1] + shape)
        
    if ft == FileType.ANGLES:
        raw_data = raw_data / 180 * np.pi # Convert to radians
        raw_data = raw_data * 2 # 'Scale' angles
        np.save(save_file, raw_data)
        
        print("Done processing angles of subject " + sub)
    elif ft == FileType.DATA:
        data = group(raw_data)
        data += np.random.normal(scale=0.5, size=data.shape) # Prevent division by zero errors
        np.save(save_file, data)

#         if __name__ == '__main__':
#             sigma = wolff_cross.prepare_sigma(data)

#         np.save(save_sigma, sigma)

        print("Done processing one data set of subject " + sub)
        
    
    if os.access(path, os.W_OK | os.X_OK):
        for file in parts:
            os.remove(file)
        print("The original data files have been removed")
    else:
        print("No proper access to " + path + ", files will not be removed")

def load_file(file):
    while True:
        try:
            arr = np.load(file)
            break
        except (OSError, ValueError) as e:
            print(str(e))
            print("Error reading file, trying again...")
            time.sleep(1)
            continue
    
    return arr

def get_dict(file):
    if neuro_pat1.search(file):
        return available_neuro1
    elif neuro_pat2.search(file):
        return available_neuro2
    elif angle_pat.search(file):
        return available_angle
    else:
        return None

class DataHandler(watchdog.events.FileSystemEventHandler):
    def __init__(self):
        super(DataHandler, self).__init__()
        self.raw_files = []
        self.processes = []
    
    def on_created(self, event):
        if not event.is_directory:
            self.raw_files.append(event.src_path)

if __name__ == "__main__":
    processed_files = set()
    processes = []
    new_files = [join(path, f) for f in os.listdir(path) if os.path.isfile(join(path, f))]
    new_files = sorted(new_files)
    
    event_handler = DataHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)

            raw_files = event_handler.raw_files
            new_files = new_files + [path for path in raw_files if path not in processed_files]

            if new_files:
                processed_files.update(new_files)
                print(new_files)

                for file in new_files:
                    # The file name has the following syntax:
                    # subj_[subject number]_[type of data]_[part_number].npy
                    # where [type of data] may contain multiple underscores
                    available_dic = get_dict(file)
                    
                    if available_dic is None:
                        continue
                    
                    sub = file.split('_')[5]
                    if sub not in available_dic:
                        available_dic[sub] = set()

                    available_dic[sub].add(file)
                    
                    if len(available_dic[sub]) == NUM_PARTS:
                        process_files(available_dic, sub)
                        
            new_files = []
            
    except KeyboardInterrupt:
        observer.stop()
        for p in event_handler.processes:
            p.join()
            
    observer.join()