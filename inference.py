import torch, os, torchaudio
import pickle
from EaBNet import make_eabnet_with_postnet
from train_distributed import prepare_data
from dataset.mcse_dataset import generate_random_noisy_for_speech, load_audio_and_random_crop
import json
import os
from scipy.io import wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
import IPython
import soundfile as sf
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if __name__ == '__main__':
    # Load the model
    args_path = 'args.pickle'
    with open(args_path, 'rb') as file:
        args = pickle.load(file)
    ckp_path = '351864.pth'
    model_ckp = torch.load(ckp_path, map_location='cpu')['model_state_dict']
    model = make_eabnet_with_postnet(args)
    model.load_state_dict(model_ckp, strict=True)

    audio_name = '1m'
    audio_dir = './demo'
    audio_path = os.path.join(audio_dir, audio_name+'.wav')
    
    model = model.cuda()
    
    noisy, sr = torchaudio.load(audio_path)
    if sr != 16000:
        noisy = torchaudio.transforms.Resample(sr, 16000)(noisy)
    
    #permute mics
    #indices = torch.tensor([3,4,5,6,7,0,1,2])
    indices = torch.tensor([7,0,1,2,3,4,5,6])
    noisy = noisy.index_select(0, indices)
    noisy = noisy.unsqueeze(0)

    noisy_stft, _ = prepare_data(noisy, noisy[:,0,...], 'cuda', args)


    with torch.no_grad():
        output = model(noisy_stft)
    
        device = 'cuda'
        esti_stft=output['esti_stft']
        print(esti_stft.shape)
        sr = args.sr
        wav_len = int(args.wav_len * sr)
        win_size = int(args.win_size * sr)
        win_shift = int(args.win_shift * sr)
        fft_num = args.fft_num
        esti_stft = esti_stft.permute(0, 3, 2, 1)
        print(esti_stft.shape)
        esti_wav = torch.istft(torch.view_as_complex(esti_stft.contiguous()), fft_num, win_shift, win_size, torch.hann_window(win_size).to(device))
        esti_wav = esti_wav.cpu().numpy()   #[1, 76640]
        wavfile.write(f'{audio_name}_out.wav', 16000, esti_wav[0])
        #show_audio(esti_wav[0], 16000,'esti_wav')