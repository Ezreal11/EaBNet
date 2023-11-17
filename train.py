import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
from torchvision import transforms
import torch.utils.data as utils
import pickle

from EaBNet import EaBNet, numParams, com_mag_mse_loss
from dataset.custom_dataset import CustomAudioVisualDataset
from torch.utils.tensorboard import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter

def load_dataset(args):
    #LOAD DATASET
    print ('\nLoading dataset')

    with open(args.training_predictors_path, 'rb') as f:
        training_audio_predictors = pickle.load(f)
    with open(args.training_target_path, 'rb') as f:
        training_target = pickle.load(f)
    with open(args.validation_predictors_path, 'rb') as f:
        validation_audio_predictors = pickle.load(f)
    with open(args.validation_target_path, 'rb') as f:
        validation_target = pickle.load(f)
    with open(args.test_predictors_path, 'rb') as f:
        test_predictors = pickle.load(f)
    with open(args.test_target_path, 'rb') as f:
        test_target = pickle.load(f)
    
    training_img_predictors = training_audio_predictors[1]
    training_audio_predictors = np.array(training_audio_predictors[0])
    training_target = np.array(training_target)
    validation_img_predictors = validation_audio_predictors[1]
    validation_audio_predictors = np.array(validation_audio_predictors[0])
    # validation_img_predictors = validation_predictors[1]
    validation_target = np.array(validation_target)
    test_audio_predictors = np.array(test_predictors[0])
    test_img_predictors = test_predictors[1]
    test_target = np.array(test_target)

    print ('\nShapes:')
    print ('Training predictors: ', training_audio_predictors.shape)
    print ('Validation predictors: ', validation_audio_predictors.shape)
    print ('Test predictors: ', test_audio_predictors.shape)

    #convert to tensor
    training_audio_predictors = torch.tensor(training_audio_predictors).float()
    validation_audio_predictors = torch.tensor(validation_audio_predictors).float()
    test_audio_predictors = torch.tensor(test_audio_predictors).float()
    training_target = torch.tensor(training_target).float()
    validation_target = torch.tensor(validation_target).float()
    test_target = torch.tensor(test_target).float()
    #build dataset from tensors
    # tr_dataset = utils.TensorDataset(training_predictors, training_target)
    # val_dataset = utils.TensorDataset(validation_predictors, validation_target)
    # test_dataset = utils.TensorDataset(test_predictors, test_target)
    
    transform = transforms.Compose([  
        transforms.ToTensor(),
    ])

    tr_dataset = CustomAudioVisualDataset((training_audio_predictors, training_img_predictors), training_target, args.path_images, args.path_csv_images_train, transform)
    #val_dataset = CustomAudioVisualDataset((validation_audio_predictors,validation_img_predictors), validation_target, args.path_images, args.path_csv_images_train, transform)
    #test_dataset = CustomAudioVisualDataset((test_audio_predictors,test_img_predictors), test_target, args.path_images, args.path_csv_images_test, transform)
    
    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
    #val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
    #test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)
    return tr_data  #, val_data, test_data

def main(args):
    if args.fixed_seed:
        seed = 1
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    net = EaBNet(k1=args.k1,
                 k2=args.k2,
                 c=args.c,
                 M=args.M,
                 embed_dim=args.embed_dim,
                 kd1=args.kd1,
                 cd1=args.cd1,
                 d_feat=args.d_feat,
                 p=args.p,
                 q=args.q,
                 is_causal=args.is_causal,
                 is_u2=args.is_u2,
                 bf_type=args.bf_type,
                 topo_type=args.topo_type,
                 intra_connect=args.intra_connect,
                 norm_type=args.norm_type,
                 ).cuda()
    net.train()
    print("The number of trainable parameters is:{}".format(numParams(net)))
    from ptflops.flops_counter import get_model_complexity_info
    #get_model_complexity_info(net, (101, 161, 9, 2))

    batch_size = args.batch_size
    mics = args.mics
    sr = args.sr
    wav_len = int(args.wav_len * sr)
    win_size = int(args.win_size * sr)
    win_shift = int(args.win_shift * sr)
    fft_num = args.fft_num
    dataloader = torch.utils.data.DataLoader(
        dataset=torch.randn(args.batch_size, wav_len, args.mics),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    dataloader = load_dataset(args)

    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

    #training loop
    for epoch in range(args.total_epoch):
        for i, data in enumerate(dataloader):
            current_iter = epoch * len(dataloader) + i

            optimizer.zero_grad()
            noisy_wav = data.cuda()
            target_wav = torch.rand(args.batch_size, wav_len).cuda()

            noisy_wav = noisy_wav.transpose(-2, -1).contiguous().view(batch_size*mics, wav_len) #[batch_size*mics, wav_len]
            
            noisy_stft = torch.stft(noisy_wav, fft_num, win_shift, win_size, torch.hann_window(win_size).to(noisy_wav.device))
            target_stft = torch.stft(target_wav, fft_num, win_shift, win_size, torch.hann_window(win_size).to(target_wav.device))
            _, freq_num, seq_len, _ = noisy_stft.shape
            noisy_stft = noisy_stft.view(batch_size, mics, freq_num, seq_len, -1).permute(0, 3, 2, 1, 4).cuda()
            target_stft = target_stft.permute(0, 3, 2, 1).cuda()
            # conduct sqrt power-compression
            noisy_mag, noisy_phase = torch.norm(noisy_stft, dim=-1) ** 0.5, torch.atan2(noisy_stft[..., -1], noisy_stft[..., 0])
            target_mag, target_phase = torch.norm(target_stft, dim=1) ** 0.5, torch.atan2(target_stft[:, -1, ...], target_stft[:, 0, ...])
            noisy_stft = torch.stack((noisy_mag * torch.cos(noisy_phase), noisy_mag * torch.sin(noisy_phase)), dim=-1).cuda()
            target_stft = torch.stack((target_mag * torch.cos(target_phase), target_mag * torch.sin(target_phase)), dim=1).cuda()

            esti_stft = net(noisy_stft)
            
            #calculate loss
            #l = com_mag_mse_loss(esti_stft, target_stft, frame_list)
            l = loss(esti_stft, target_stft)
            print('loss:', l.item())
            l.backward()
            optimizer.step()

            if current_iter % 100 == 0:
                print('iter:', current_iter)
                print('loss:', l.item())

            print('data:', data.shape)
            print('noisy_wav:', noisy_wav.shape)
            print('noisy_stft:', noisy_stft.shape)
            print('esti_stft:', esti_stft.shape)
            print('target_stft:', target_stft.shape)
            break
        break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("This script provides the network code and a simple testing, you can train the"
                                     "network according to your own pipeline")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--total_epoch", type=int, default=100)
    parser.add_argument("--mics", type=int, default=9)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--wav_len", type=float, default=6.0)
    parser.add_argument("--win_size", type=float, default=0.020)
    parser.add_argument("--win_shift", type=float, default=0.010)
    parser.add_argument("--fft_num", type=int, default=320)
    parser.add_argument("--k1", type=tuple, default=(2,3))
    parser.add_argument("--k2", type=tuple, default=(1,3))
    parser.add_argument("--c", type=int, default=64)
    parser.add_argument("--M", type=int, default=9)
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

    #dataset parameters
    parser.add_argument('--training_predictors_path', type=str, default='/data/wbh/l3das23/processed/task1_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str, default='/data/wbh/l3das23/processed/task1_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str, default='/data/wbh/l3das23/processed/task1_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str, default='/data/wbh/l3das23/processed/task1_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str, default='/data/wbh/l3das23/processed/task1_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default='/data/wbh/l3das23/processed/task1_target_test.pkl')
    
    args = parser.parse_args()

    main(args)