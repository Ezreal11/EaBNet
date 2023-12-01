from argparse import ArgumentParser
import glob
import os
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np
from pesq import pesq
from pystoi import stoi
from dataset import make_dataset
from metrics import energy_ratios, mean_std
from soundfile import read
from os.path import join
import resampy
import pandas as pd
from EaBNet import EaBNet
import torch
import torch.utils.data as utils
#import pysepm

def prepare_data(x, target, device, args):
    batch_size = x.shape[0]
    mics = args.mics
    sr = args.sr
    wav_len = int(args.wav_len * sr)
    win_size = int(args.win_size * sr)
    win_shift = int(args.win_shift * sr)
    fft_num = args.fft_num

    noisy_wav = x.to(device)    #[4, 4, 76672]
    #target_wav = torch.rand(args.batch_size, wav_len).cuda()
    target_wav = target.to(device)  #[4, 1, 76672]
    noisy_wav = noisy_wav.contiguous().view(batch_size*mics, -1)#noisy_wav.shape[-1]) #[batch_size*mics, wav_len]
    target_wav = target.squeeze(1)
    #[batch_size*mics, freq_num, seq_len, 2]
    noisy_stft = torch.stft(noisy_wav, fft_num, win_shift, win_size, torch.hann_window(win_size).to(noisy_wav.device),return_complex=False)
    target_stft = torch.stft(target_wav, fft_num, win_shift, win_size, torch.hann_window(win_size).to(target_wav.device),return_complex=False)
    _, freq_num, seq_len, _ = noisy_stft.shape
    noisy_stft = noisy_stft.view(batch_size, mics, freq_num, seq_len, -1).permute(0, 3, 2, 1, 4).to(device)
    target_stft = target_stft.permute(0, 3, 2, 1).to(device)
    # conduct sqrt power-compression
    noisy_mag, noisy_phase = torch.norm(noisy_stft, dim=-1) ** 0.5, torch.atan2(noisy_stft[..., -1], noisy_stft[..., 0])
    target_mag, target_phase = torch.norm(target_stft, dim=1) ** 0.5, torch.atan2(target_stft[:, -1, ...], target_stft[:, 0, ...])
    noisy_stft = torch.stack((noisy_mag * torch.cos(noisy_phase), noisy_mag * torch.sin(noisy_phase)), dim=-1).to(device)
    target_stft = torch.stack((target_mag * torch.cos(target_phase), target_mag * torch.sin(target_phase)), dim=1).to(device)
    
    #noisy: [4, 601, 161, 9, 2]
    return noisy_stft, target_stft


def cal_metrics(gt_paths, noisy_paths, output_paths):
    pesqs = []
    nb_pesqs = []
    stois = []
    sisdrs = []
    data = {"filename": [], "pesq":[], "nb_pesq": [],"stoi":[],  "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": [], "csig": [], "cbak": [], "covl": []}

    
    for i in tqdm(range(len(output_paths))):
        name1, name2, name3 = os.path.basename(gt_paths[i]), os.path.basename(output_paths[i]), os.path.basename(noisy_paths[i])
        assert(name1 == name2 == name3)
        
        #-------------------------读取、重采样、对齐-------------------------
        sr = 16000
        x_method, _ = read(output_paths[i])
        x, _ = read(gt_paths[i])
        x = resampy.resample(x, _ , 16000, axis=0)
        y, _ = read(noisy_paths[i])
        y = resampy.resample(y, _ , 16000, axis=0)
        maxlen = max(x.shape[0], y.shape[0], x_method.shape[0])
        #align
        audios = [x, y, x_method]
        for index, audio in enumerate(audios):
            if audio.shape[0] != maxlen:
                new = np.zeros((maxlen,) + audio.shape[1:])
                new[:audio.shape[0]] = audio
                audios[index] = new
        x, y, x_method = audios
        n = y - x 
        
        data["filename"].append(name1)
        data["pesq"].append(pesq(sr, x, x_method, 'wb'))
        data["nb_pesq"].append(pesq(sr, x, x_method,'nb'))
        data["stoi"].append(stoi(x, x_method, sr, extended=False))
        data["estoi"].append(stoi(x, x_method, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_method, x, n)[0])
        data["si_sir"].append(energy_ratios(x_method, x, n)[1])
        data["si_sar"].append(energy_ratios(x_method, x, n)[2])

        '''CSIG, CBAK, COVL = pysepm.composite(x, x_method, sr)
        data["csig"].append(CSIG)
        data["cbak"].append(CBAK)
        data["covl"].append(COVL)'''
        #================= 


    df = pd.DataFrame(data)
    df.to_csv(score_file, index=False)
    f = open(txt_file, 'w')
    f.write("PESQ: {:.2f} ± {:.2f}\n".format(*mean_std(df["pesq"].to_numpy())))
    f.write("NBPESQ: {:.2f} ± {:.2f}\n".format(*mean_std(df["nb_pesq"].to_numpy())))
    f.write("STOI: {:.2f} ± {:.2f}\n".format(*mean_std(df["stoi"].to_numpy())))
    f.write("ESTOI: {:.2f} ± {:.2f}\n".format(*mean_std(df["estoi"].to_numpy())))
    f.write("SI-SDR: {:.2f} ± {:.2f}\n".format(*mean_std(df["si_sdr"].to_numpy())))
    f.write("SI-SIR: {:.2f} ± {:.2f}\n".format(*mean_std(df["si_sir"].to_numpy())))
    f.write("SI-SAR: {:.2f} ± {:.2f}\n".format(*mean_std(df["si_sar"].to_numpy())))
    f.write("CSIG: {:.2f} ± {:.2f}\n".format(*mean_std(df["csig"].to_numpy())))
    f.write("CBAK: {:.2f} ± {:.2f}\n".format(*mean_std(df["cbak"].to_numpy())))
    f.write("COVL: {:.2f} ± {:.2f}\n".format(*mean_std(df["covl"].to_numpy())))
    f.close()
    
    # Print results
    #print("POLQA: {:.2f} ± {:.2f}".format(*mean_std(df["polqa"].to_numpy())))
    print(score_file)
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("NBPESQ: {:.2f} ± {:.2f}".format(*mean_std(df["nb_pesq"].to_numpy())))
    print("STOI: {:.2f} ± {:.2f}".format(*mean_std(df["stoi"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.2f} ± {:.2f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.2f} ± {:.2f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.2f} ± {:.2f}".format(*mean_std(df["si_sar"].to_numpy())))
    print("CSIG: {:.2f} ± {:.2f}\n".format(*mean_std(df["csig"].to_numpy())))
    print("CBAK: {:.2f} ± {:.2f}\n".format(*mean_std(df["cbak"].to_numpy())))
    print("COVL: {:.2f} ± {:.2f}\n".format(*mean_std(df["covl"].to_numpy())))
    

def cal_single_metrics(gt, y, x_method, sr=16000):
    '''inputs are all numpy arrays'''
    ret = {}
    maxlen = max(gt.shape[0], y.shape[0], x_method.shape[0])
    #align
    audios = [gt, y, x_method]
    for index, audio in enumerate(audios):
        if audio.shape[0] != maxlen:
            new = np.zeros((maxlen,) + audio.shape[1:])
            new[:audio.shape[0]] = audio
            audios[index] = new
    gt, y, x_method = audios
    n = y - gt 

    ret["pesq"] = pesq(sr, gt, x_method, 'wb')
    ret["nb_pesq"] = pesq(sr, gt, x_method,'nb')
    ret["stoi"] = stoi(gt, x_method, sr, extended=False)
    ret["estoi"] = stoi(gt, x_method, sr, extended=True)
    ret["si_sdr"] = energy_ratios(x_method, gt, n)[0]
    ret["si_sir"] = energy_ratios(x_method, gt, n)[1]
    ret["si_sar"] = energy_ratios(x_method, gt, n)[2]

    '''CSIG, CBAK, COVL = pysepm.composite(gt, x_method, sr)
    ret["csig"] = CSIG
    ret["cbak"] = CBAK
    ret["covl"] = COVL'''
    
    return ret

@torch.no_grad()
def test(args):
    # Load model
    print("Loading model...")
    device = torch.device('cuda:{}'.format(0))
    net = EaBNet(k1=args.k1, k2=args.k2, c=args.c, M=args.M, embed_dim=args.embed_dim, kd1=args.kd1, cd1=args.cd1,
                 d_feat=args.d_feat, p=args.p, q=args.q, is_causal=args.is_causal, is_u2=args.is_u2, bf_type=args.bf_type,
                 topo_type=args.topo_type, intra_connect=args.intra_connect, norm_type=args.norm_type,).to(device)
    checkpoint_path = args.model_path
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print(f"Model loaded from {checkpoint_path}")
    
    tr_dataset, val_dataset = make_dataset(args)
    valloader = utils.DataLoader(val_dataset, 1, shuffle=False)

    data = {"filename": [], "pesq":[], "nb_pesq": [],"stoi":[],  "estoi": [], "si_sdr": [], "si_sir": [],
              "si_sar": [], "csig": [], "cbak": [], "covl": []}

    for i, (x, target) in enumerate(tqdm(valloader,desc=f'Valid:')):
        target = target.to(device)
        x = x.to(device)
        noisy_stft, target_stft = prepare_data(x, target, device, args)
        esti_stft = net(noisy_stft)
        #loss = criterion(esti_stft, target_stft)
        frame_list = [noisy_stft.shape[1]] * x.shape[0]
        #loss = com_mag_mse_loss(esti_stft, target_stft, frame_list)
        #loss_list.append(loss.item()/torch.distributed.get_world_size())
        sr = args.sr
        wav_len = int(args.wav_len * sr)
        win_size = int(args.win_size * sr)
        win_shift = int(args.win_shift * sr)
        fft_num = args.fft_num
        esti_stft, target_stft = esti_stft.permute(0, 3, 2, 1), target_stft.permute(0, 3, 2, 1)
        esti_wav = torch.istft(torch.view_as_complex(esti_stft.contiguous()), fft_num, win_shift, win_size, torch.hann_window(win_size).to(device))
        esti_wav = esti_wav.cpu().numpy()   #[1, 76640]
        noisy_wav = x.squeeze(0).cpu().numpy()  #[4, 76672]
        target_wav = target.squeeze(0).cpu().numpy()    #[1, 76672]
        ret = cal_single_metrics(target_wav[0], noisy_wav[0], esti_wav[0], sr)
        for k, v in ret.items():
            data[k].append(v)

        
    
    new_data = {}
    for k, v in data.items():
        if len(v) > 0:
            new_data[k] = v
    data = new_data
    score_path = os.path.join(os.path.dirname(args.model_path), "score.txt")
    df = pd.DataFrame(data)
    with open(score_path, 'w') as f:
        for k, v in data.items():
            print("{}: {:.2f} ± {:.2f}".format(k, *mean_std(np.array(v))))
            f.write("{}: {:.2f} ± {:.2f}\n".format(k, *mean_std(np.array(v))))
            

if __name__ == '__main__':
    parser = ArgumentParser(description='caluculate metrics')
    parser.add_argument('--model_path',default="/data/wbh/l3das23/experiment/4gpu/28188.pth", type=str)
    #eabnet parameters
    parser.add_argument("--batch_size", type=int, default=6)    #8 in paper
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--total_epoch", type=int, default=100)  #60 in paper
    parser.add_argument("--mics", type=int, default=8)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--wav_len", type=float, default=6.0)
    parser.add_argument("--win_size", type=float, default=0.020)
    parser.add_argument("--win_shift", type=float, default=0.010)
    parser.add_argument("--fft_num", type=int, default=320)
    parser.add_argument("--k1", type=tuple, default=(2,3))
    parser.add_argument("--k2", type=tuple, default=(1,3))
    parser.add_argument("--c", type=int, default=64)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--kd1", type=int, default=5)
    parser.add_argument("--cd1", type=int, default=64)
    parser.add_argument("--d_feat", type=int, default=256)
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--is_causal", type=bool, default=True, choices=[True, False])
    parser.add_argument("--is_u2", type=bool, default=True, choices=[True, False])
    parser.add_argument("--bf_type", type=str, default="lstm", choices=["lstm", "cnn"])
    parser.add_argument("--topo_type", type=str, default="mimo", choices=["mimo", "miso"])
    parser.add_argument("--intra_connect", type=str, default="cat", choices=["cat", "add"])
    parser.add_argument("--norm_type", type=str, default="IN", choices=["BN", "IN", "cLN"])
    parser.add_argument("--fixed_seed", type=bool, default=False, choices=[True, False])

    parser.add_argument('--dataset', type=str, default='mcse', choices=['l3das23', 'mcse'])
    parser.add_argument('--mcse_dataset_train_speech_root', type=str, default='data/datasets/datasets_fullband/clean_fullband/read_speech')
    parser.add_argument('--mcse_dataset_train_noise_root', type=str, default='data/datasets/datasets_fullband/noise_fullband')
    parser.add_argument('--mcse_dataset_train_set', type=str, choices=['online','offline'], default='online')
    args = parser.parse_args()

    test(args)