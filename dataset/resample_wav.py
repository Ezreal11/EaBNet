import os
from tqdm import tqdm
from scipy.io import wavfile
import scipy.signal as signal

def proc_audio(filename,tgt_filename, resample_fs=16000):
    fs, audio = wavfile.read(filename)
    if fs!=resample_fs:
        audio = signal.resample(audio, int(resample_fs*audio.shape[-1]/fs)).astype(audio.dtype)
    wavfile.write(tgt_filename, resample_fs, audio)

def proc_dir(src_dir,tgt_dir,resample_fs=16000):
    print(f'{src_dir} -> {tgt_dir}')
    if not os.path.isdir(tgt_dir):
        os.makedirs(tgt_dir)
    src_audios = os.listdir(src_dir)
    for audio in tqdm(src_audios):
        proc_audio(os.path.join(src_dir,audio),os.path.join(tgt_dir,audio),resample_fs)

if __name__ == '__main__':
    proc_dir('/data1/zhouchang/datasets/datasets_fullband/noise_fullband',
             '/data1/zhouchang/datasets/datasets_16khz/noise_16khz')
    proc_dir('/data1/zhouchang/datasets/datasets_fullband/clean_fullband/read_speech',
             '/data1/zhouchang/datasets/datasets_16khz/clean_16khz/read_speech')
