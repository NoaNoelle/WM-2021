import time
import traceback
import watchdog
from watchdog.observers import Observer
import multiprocessing as mp
import numpy as np
import re
import os
from os.path import join, exists
from enum import Enum, auto

import wolff_cross

path = '/Users/s3344282/Analysis/full_model/exp2_content_specific_reactivation_model'
device_i = 1

# Only decode the first 1200 ms
decode_up_to = 600

def load_file(file, mmap=None):
    while True:
        try:
            arr = np.load(file, mmap_mode=mmap)
            break
        except (OSError, ValueError) as e:
            print(str(e))
            print("Error reading file, trying again...")
            time.sleep(1)
            continue
    
    return arr

def decode_part(dat_range, path, module, path_sigma, save_file, device_i):
    start_t = time.time()
    
    if not exists(save_file):
        print("Loading data...")
        data = load_file(join(path, "data" + module + ".npy"), 'r')
        data = data[:, :, :decode_up_to] # Decode only the first decode_up_to timesteps
        data = data[dat_range].copy()
        print("Loading angles...")
        angles = load_file(join(path, "angles.npy"), 'r')
        angles = angles[dat_range].copy()
        print("Loading sigma...")
        sigma = load_file(path_sigma)
        print("All files loaded")
        
        bin_width = np.pi / 6

        print("Decoding...")
        cross_cos_amp = wolff_cross.cross_decode(data, angles, bin_width, sigma, device_i)
#         cross_cos_amp = wolff_cross.cross_decode(data, angles, bin_width, sigma, 2)

        c = np.mean(cross_cos_amp, 0)

        np.save(save_file, c)
    
    print("Done with " + save_file)
    end_t = time.time()
    
    with open(join(path, "diagnostics.txt"), 'a') as f:
        f.write(str(end_t - start_t) + "\n")

def calc_sigmas(len_i, split_i, path, module):
    # data shape: trials by channels by timesteps
    data = load_file(join(path, "data" + module + ".npy"), 'r')
    
    # Decode only the first decode_up_to timesteps
    data = data[:, :, :decode_up_to]
    
    data0 = data[range(split_i)].copy()
    data1 = data[range(split_i, len_i)].copy()
    
    path0 = join(path, "sigma" + module + "_0.npy")
    path1 = join(path, "sigma" + module + "_1.npy")
    
    if not exists(path0):
        sigma0 = wolff_cross.prepare_sigma(data0)
        np.save(path0, sigma0)
    
    if not exists(path1):
        sigma1 = wolff_cross.prepare_sigma(data1)
        np.save(path1, sigma1)
    
    return (path0, path1)
    
def err_handler(e):
    print(str(e))
    traceback.print_tb(e.__traceback__)
    
def decode_file(pool, path, module):
    save_file = "c" + module
    save_file = join(path, save_file)
    
    print("Starting with " + path + ", module " + module)

    # Split up all data evenly
    angles = load_file(join(path, "angles.npy"), 'r') # We have to load this to check its length
    len_i = len(angles)
    split_i = int(len_i / 2)
    
    path_sigma0, path_sigma1 = calc_sigmas(len_i, split_i, path, module)
    
    global device_i

    pool.apply_async(decode_part, 
                     (range(split_i), path, module, path_sigma0, save_file+"_0.npy", device_i), 
                     error_callback=err_handler)
    device_i = (device_i + 1) % 4
    
    pool.apply_async(decode_part, 
                     (range(split_i, len_i), path, module, path_sigma1, save_file+"_1.npy", device_i), 
                     error_callback=err_handler)
    device_i = (device_i + 1) % 4

def get_present_files():
    files = []
    pat = re.compile(r"(data)|(sigma)|(angles)")
    
    for (dirpath, dirnames, filenames) in os.walk(path):
        files += ([join(dirpath, f) for f in filenames if pat.search(f)])
        
    return sorted(files)
        
class PrepDataHandler(watchdog.events.FileSystemEventHandler):
    def __init__(self):
        super(PrepDataHandler, self).__init__()
        self.files = []
    
    def on_created(self, event):
        if not event.is_directory:
            self.files.append(event.src_path)

if __name__ == "__main__":
    processed_files = set()
    new_files = get_present_files()
    
    event_handler = PrepDataHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        with mp.Pool(4) as pool:
            while True:
                time.sleep(4)

                new_files = new_files + [path for path in event_handler.files if path not in processed_files]
                
                if new_files:
                    print("new_files:", new_files)
                    print("processed_files:", processed_files)
                    
                    for file in new_files:
                        f_dir, f_name = os.path.split(file)
                        
                        print(file)
                        
                        if re.search(r"angles.npy", f_name):
                            if join(f_dir, "data1.npy") in processed_files:
                                print("Everything present for first module")
                                decode_file(pool, f_dir, "1")
                                
                            if join(f_dir, "data2.npy") in processed_files:
                                print("Everything present for second module")
                                decode_file(pool, f_dir, "2")
                        
                        if re.search(r"data", f_name) or re.search(r"sigma", f_name):
                            module = re.sub(r"(data)|(sigma)|(\.npy)", "", f_name)
                            dat_file = "data" + module + ".npy"
                            sig_file = "sigma" + module + ".npy"
                            
                            deps = join(f_dir, "angles.npy") in processed_files
                            print("Angles present:", deps)
                            dat_dep = f_name == dat_file or join(f_dir, dat_file) in processed_files
#                             sig_dep = f_name == sig_file or join(f_dir, sig_file) in processed_files
                            sig_dep = True
                            deps = deps and dat_dep and sig_dep
        
                            print("Data present:", dat_dep)
                            print("Sigma present:", sig_dep)
                            
                            if deps:
                                decode_file(pool, f_dir, module)
                            
                        processed_files.add(file)
                
                new_files = []
    except KeyboardInterrupt:
        observer.stop()
    observer.join()