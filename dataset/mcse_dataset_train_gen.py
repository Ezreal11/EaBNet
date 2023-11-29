from tqdm import tqdm
import os
import json
import numpy as np
from scipy.io import wavfile
import sys
sys.path.append('..')
from mcse_dataset import generate_random_noisy_for_speech
import multiprocessing as mp

data_train_root = '../data/datasets/mcse_train'
if not os.path.exists(data_train_root):
    os.makedirs(data_train_root)
    
noisy_train_root = os.path.join(data_train_root,'noisy')
if not os.path.exists(noisy_train_root):
    os.makedirs(noisy_train_root)
    
clean_train_root = os.path.join(data_train_root,'clean')
if not os.path.exists(clean_train_root):
    os.makedirs(clean_train_root)
    
def write_audio(audio, fs, path):
    audio = np.clip(audio,-1,1)*np.iinfo(np.int16).max-1
    wavfile.write(path, fs, audio.astype(np.int16))


with open('../data/datasets/datasets_fullband/cleans_train','r') as f:
    cleans_train = f.read().split('\n')

with open('../data/datasets/datasets_fullband/noises_train','r') as f:
    noises_train = f.read().split('\n')

with open('mcse_dataset_settings.json','r') as f:
    opt = json.load(f)

clean_root = '../data/datasets/datasets_fullband/clean_fullband/read_speech'
noise_root = '../data/datasets/datasets_fullband/noise_fullband'
fs = 16000
clip_seconds = 6

def worker(i):
    global clean_root, noise_root, fs, clip_seconds, idx, noisy_train_root, clean_train_root, cleans_train, noises_train
    clean = cleans_train[i]
    clean_fs, clean_audio = wavfile.read(os.path.join(clean_root,clean))
    t = 0
    j = 0
    while (t+clip_seconds)*clean_fs <= len(clean_audio):
        sample = generate_random_noisy_for_speech(opt, clip_seconds, clean, noises_train, clean_root, noise_root, speech_start_sec=t)
        name = f'{i:05}_{j}'
        j += 1
        write_audio(sample['noisy'].T,fs,os.path.join(noisy_train_root,f'{name}.wav'))
        write_audio(sample['clean'],fs,os.path.join(clean_train_root,f'{name}.wav'))
        t += clip_seconds

pool = mp.Pool(16)
idx = range(1,len(cleans_train),3)
with tqdm(total=len(idx)) as pbar:
    for i in idx:
        pool.apply_async(worker, args=(i,), callback=lambda x: pbar.update())
    pool.close()
    pool.join()
