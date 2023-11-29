import os
import torch
import torchaudio
import torch.utils.data as data
import pyroomacoustics as pra
import numpy as np
from scipy.io import wavfile
from dataset.audio_util import make_audio
import scipy.signal as signal
import json


def random_float(bounds):
    return bounds[0] + (bounds[1]-bounds[0])*np.random.random()

def cal_angle(v1,v2):
    angle_rad = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def cal_rotate_matrix_2d(v,v_tgt):
    abs_angle = cal_angle(v,v_tgt)
    is_clock_wise = np.cross(v,v_tgt) < 0
    angle = -abs_angle if is_clock_wise else abs_angle
    rad = np.radians(angle)
    R = np.array([
        [np.cos(rad),-np.sin(rad)],
        [np.sin(rad),np.cos(rad)]
    ])
    return R

def load_audio_and_random_crop(filename,resample_fs, crop_seconds, start_seconds=None):
    fs, audio = wavfile.read(filename)
    n_points = fs*crop_seconds
    if len(audio) < n_points:
        audio = np.append(audio, np.zeros(n_points-len(audio)))
    if start_seconds is None:
        start = np.random.randint(0, len(audio)-n_points+1)
    else:
        start = start_seconds*fs
    audio = audio[start:start+n_points]
    if resample_fs != fs:
        audio = signal.resample(audio, resample_fs*crop_seconds).astype(audio.dtype)
    return audio


def generate_random_noisy_for_speech(opt, clip_seconds, target_speech, all_noises, speech_root, noise_root, speech_start_sec=None):

    # generate random room

    min_dim = np.array(opt['room']['min_dim'])
    max_dim = np.array(opt['room']['max_dim'])
    room_dim = min_dim + (max_dim-min_dim)*np.random.random([3])

    # load mics

    p_mics = np.array([[mic['x'],mic['y']] for mic in opt['mic_array']['mics']]) # 8,2
    p_mics = p_mics.T # 2, n_mics
    direction_mics = opt['mic_array']['direction']
    direction_mics = np.array([direction_mics['x'],direction_mics['y']]) # 2

    # generate random target and mics position

    fail_count = 0

    while True:
        d = opt['target']['min_dist_to_wall']
        target_x = random_float([d,room_dim[0]-d])
        target_y = random_float([d,room_dim[1]-d])
        target_z = random_float(opt['target']['h'])

        d = opt['mic_array']['min_dist_to_wall']
        mics_x = random_float([d,room_dim[0]-d])
        mics_y = random_float([d,room_dim[1]-d])
        mics_z = random_float(opt['mic_array']['h'])

        dist = sum([o*o for o in [target_x-mics_x, target_y-mics_y, target_z-mics_z]])**0.5
        dist_bounds = opt['target']['dist_to_mic_array']
        if dist<dist_bounds[0] or dist>dist_bounds[1]:
            fail_count += 1
            continue
        break

    p_target = np.array([target_x,target_y,target_z])
    p_target_2d = p_target[:2]
    p_mics_cen = np.array([mics_x,mics_y,mics_z])
    p_mics_cen_2d = p_mics_cen[:2]

    # rotate the mic array to make its direction is toward the target
    assert opt['target']['fixed_doa'], 'Not supported'
    direction_target = p_target_2d - p_mics_cen_2d
    R = cal_rotate_matrix_2d(direction_mics, direction_target)
    direction_mics = np.dot(R, direction_mics)
    p_mics = np.dot(R, p_mics) # 2, n_mics
    p_mics = np.concatenate([p_mics, np.zeros((1,p_mics.shape[1]))],0) # 3, n_mics
    p_mics = p_mics + p_mics_cen.reshape((3,1))

    # generate random noises position
    n_noises = opt['noise']['n']
    n_noises = np.random.randint(n_noises[0],n_noises[1]+1)
    p_noise_list = []
    noise_list = np.random.choice(all_noises, n_noises)
    snr_list = []

    for i in range(n_noises):
        while True:
            x = random_float([0,room_dim[0]])
            y = random_float([0,room_dim[1]])
            z = random_float(opt['noise']['h'])
            dist = sum([o*o for o in [x-mics_x, y-mics_y, z-mics_z]])**0.5
            if dist < opt['noise']['min_dist_to_mic_array']:
                fail_count += 1
                continue
            p_noise = np.array([x,y,z])
            ang = cal_angle(p_target-p_mics_cen,p_noise-p_mics_cen)
            if ang < opt['noise']['min_doa_diff_wrt_target']:
                fail_count += 1
                continue
            break
        p_noise_list.append(p_noise)
        snr_list.append(random_float(opt['noise']['SNR']))

    # generate random rt60
    while True:
        rt60_tgt = random_float(opt['room']['rt60'])
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
        except ValueError:
            # room too large for given rt60
            fail_count += 1
            continue
        break

    if fail_count >= 50:
        print(f'Random position generation failed {fail_count} times in a sample, the restriction may be too tight')

    noisy_dBFS = random_float(opt['noisy_dBFS'])

    fs = opt['audio']['fs']

    audio_clean = load_audio_and_random_crop(os.path.join(speech_root, target_speech), resample_fs=fs, crop_seconds=clip_seconds, start_seconds=speech_start_sec)
    audio_noises = []
    for x in noise_list:
        audio_noises.append(load_audio_and_random_crop(os.path.join(noise_root,x), resample_fs=fs, crop_seconds=clip_seconds))


    meta = {
        'room_dim': room_dim,
        'e_absorption': e_absorption,
        'max_order': max_order,
        'fs': fs,
        'p_mics': p_mics,
        'p_target': p_target,
        'p_noise_list': p_noise_list,
        'snr_list': snr_list,
        'dBFS': noisy_dBFS,
        'clean': target_speech,
        'noises': noise_list
    }

    room, freefield, clean, noisy = make_audio(
        room_dim=room_dim,
        e_absorption=e_absorption,
        max_order=max_order,
        rir_method=opt['audio']['rir_method'],
        fs=fs,
        ref_mic=opt['mic_array']['ref_mic'],
        p_mics=p_mics,
        p_target=p_target,
        p_noise_list=p_noise_list,
        snr_noises=snr_list,
        dBFS=noisy_dBFS,
        clean=audio_clean,
        noises=audio_noises
    )

    return {
        'meta':meta, 
        'room':room, 
        'freefield':freefield, 
        'clean': clean, 
        'noisy': noisy 
    }

class McseDatasetOnline(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.speech_root = opt['speech_root']
        self.noise_root = opt['noise_root']
        with open(opt['mcse_settings'],'r') as f:
            self.mcse_settings = json.load(f)
        with open(opt['speech_list'],'r') as f:
            self.speech_list = f.read().split('\n')
        with open(opt['noise_list'],'r') as f:
            self.noise_list = f.read().split('\n')
        self.clip_seconds = opt['clip_seconds']
        
    def __len__(self):
        return len(self.speech_list)

    def __getitem__(self, index):
        sample = generate_random_noisy_for_speech(
            opt=self.mcse_settings, 
            clip_seconds=self.clip_seconds,
            target_speech=self.speech_list[index],
            all_noises=self.noise_list,
            speech_root=self.speech_root,
            noise_root=self.noise_root
            )
        clean = sample['clean']
        noisy = sample['noisy']
        return torch.tensor(noisy,dtype=torch.float), torch.tensor(clean,dtype=torch.float).reshape(1,-1)


class McseDatasetOffline(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.clean_root = opt['clean_root']
        self.noisy_root = opt['noisy_root']
        self.sample_list = os.listdir(self.clean_root)
        self.sample_list.sort()
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample = self.sample_list[index]
        clean_path = os.path.join(self.clean_root, sample)
        noisy_path = os.path.join(self.noisy_root, sample)
        clean, _ = torchaudio.load(clean_path)
        noisy, _ = torchaudio.load(noisy_path)
        return noisy, clean
        

def make_mcse_dataset(args):
    if args.mcse_dataset_train_set == 'online':
        train_dataset = McseDatasetOnline({
            'speech_root': args.mcse_dataset_train_speech_root,
            'noise_root': args.mcse_dataset_train_noise_root,
            'speech_list': 'data/datasets/datasets_fullband/cleans_train',
            'noise_list': 'data/datasets/datasets_fullband/noises_train',
            'mcse_settings': 'dataset/mcse_dataset_settings.json',
            'clip_seconds': 6
        })
    elif args.mcse_dataset_train_set == 'offline':
        train_dataset = McseDatasetOffline({
            'clean_root': 'data/datasets/mcse_train/clean',
            'noisy_root': 'data/datasets/mcse_train/noisy'
        })
    val_dataset = McseDatasetOffline({
        'clean_root': 'data/datasets/mcse_val/clean',
        'noisy_root': 'data/datasets/mcse_val/noisy'
    })
    return train_dataset, val_dataset