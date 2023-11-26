import numpy as np
import pyroomacoustics as pra

EPS = np.finfo(float).eps

def active_noise_rms(noise, fs, energy_thresh=-50):
    '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
    window_size = 100 # in ms
    window_samples = int(fs*window_size/1000)
    sample_start = 0
    noise_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        noise_seg_rms = (noise_win**2).mean()**0.5
        # Considering frames with energy
        if noise_seg_rms > 10**(energy_thresh/20):
            noise_active_segs = np.append(noise_active_segs, noise_win)
        sample_start += window_samples

    if len(noise_active_segs)!=0:
        noise_rms = (noise_active_segs**2).mean()**0.5
    else:
        noise_rms = EPS
        
    return noise_rms

def mix_scaler(clean, noises, SNRs, mixed_dBFS, fs):
    clean = clean/(max(abs(clean))+EPS)
    noises = [x/max(abs(x)+EPS) for x in noises]
    rms_clean = (clean**2).mean()**0.5
    scaled_noises = []
    for noise, snr in zip(noises,SNRs):
        rms_noise = active_noise_rms(noise, fs)
        scale = rms_clean / (10**(snr/20)) / (rms_noise+EPS)
        scaled_noises.append(noise*scale)
    noisy = clean
    for noise in scaled_noises:
        noisy = noisy + noise
    rms_noisy = (noisy**2).mean()**0.5
    scale = 10 ** (mixed_dBFS/20) / (rms_noisy+EPS)
    scaled_clean = scale * clean
    scaled_noises = [scale*x for x in scaled_noises]

    return scaled_clean, scaled_noises


def make_audio(room_dim, e_absorption, max_order, rir_method, fs, ref_mic, p_mics, p_target, p_noise_list, snr_noises, dBFS, clean, noises):

    if rir_method == "ism":
        room = pra.ShoeBox(
            room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
        )
    elif rir_method == "hybrid":
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=3,
            ray_tracing=True,
            air_absorption=True,
        )
    else:
        raise ValueError

    freefield = pra.AnechoicRoom(3, fs=fs)
    n_points = len(clean)
    clean, noises = mix_scaler(clean, noises, snr_noises, dBFS, fs)
    room.add_source(p_target, clean)
    freefield.add_source(p_target, clean)

    for i in range(len(noises)):
        room.add_source(p_noise_list[i], noises[i])
    
    room.add_microphone_array(p_mics)
    freefield.add_microphone_array(p_mics)
    
    room.simulate()
    noisy = room.mic_array.signals

    freefield.simulate()
    clean = freefield.mic_array.signals[ref_mic]

    noisy = noisy[:,:n_points]
    clean = clean[:n_points]

    return room, freefield, clean, noisy
