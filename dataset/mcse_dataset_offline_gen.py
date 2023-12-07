from tqdm import tqdm
import os
import json
import numpy as np
from scipy.io import wavfile
import sys
sys.path.append('..')
from mcse_dataset import generate_random_noisy_for_speech
import multiprocessing as mp
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--speech_root', type=str, default='../data/datasets/datasets_fullband/clean_fullband/read_speech')
parser.add_argument('--noise_root', type=str, default='../data/datasets/datasets_fullband/noise_fullband')
parser.add_argument('--speech_list', type=str, required=True)
parser.add_argument('--noise_list', type=str, required=True)
parser.add_argument('--mcse_settings', type=str, required=True)
parser.add_argument('--clip_seconds', type=int, required=True)
parser.add_argument('--reuse_speech', action='store_true', default=False)

args = parser.parse_args()

data_root = args.output_dir
if not os.path.exists(data_root):
    os.makedirs(data_root)
    
noisy_root = os.path.join(data_root,'noisy')
if not os.path.exists(noisy_root):
    os.makedirs(noisy_root)
    
clean_root = os.path.join(data_root,'clean')
if not os.path.exists(clean_root):
    os.makedirs(clean_root)
    
def write_audio(audio, fs, path):
    audio = np.clip(audio,-1,1)*np.iinfo(np.int16).max
    wavfile.write(path, fs, audio.astype(np.int16))


with open(args.speech_list,'r') as f:
    speech_list = f.read().split('\n')

with open(args.noise_list,'r') as f:
    noise_list = f.read().split('\n')

with open(args.mcse_settings,'r') as f:
    opt = json.load(f)

speech_root = args.speech_root
noise_root = args.noise_root
fs = 16000
clip_seconds = args.clip_seconds
reuse_speech = args.reuse_speech

def worker(i):
    global clean_root, noise_root, fs, clip_seconds, idx, noisy_root, speech_root, speech_list, noise_list, reuse_speech 

    speech = speech_list[i]
    if not reuse_speech:
        sample = generate_random_noisy_for_speech(opt, clip_seconds, speech, noise_list, speech_root, noise_root)
        name = f'{i:05}'
        write_audio(sample['noisy'].T,fs,os.path.join(noisy_root,f'{name}.wav'))
        write_audio(sample['clean'],fs,os.path.join(clean_root,f'{name}.wav'))
        return

    speech_fs, speech_audio = wavfile.read(os.path.join(clean_root,speech))
    t = 0
    j = 0
    while (t+clip_seconds)*speech_fs <= len(speech_audio):
        sample = generate_random_noisy_for_speech(opt, clip_seconds, speech, noise_list, speech_root, noise_root, speech_start_sec=t)
        name = f'{i:05}_{j}'
        j += 1
        write_audio(sample['noisy'].T,fs,os.path.join(noisy_root,f'{name}.wav'))
        write_audio(sample['clean'],fs,os.path.join(clean_root,f'{name}.wav'))
        t += clip_seconds

def set_seed():
    np.random.seed(os.getpid()+12345)
    print(f'p{os.getpid()} set seed')
    time.sleep(1)

n_workers = 16
pool = mp.Pool(n_workers)
    
for i in range(n_workers):
    pool.apply_async(set_seed)

idx = range(len(speech_list))
with tqdm(total=len(idx)) as pbar:
    for i in idx:
        pool.apply_async(worker, args=(i,), callback=lambda x: pbar.update())
    pool.close()
    pool.join()
