import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as utils
import pickle
from tqdm import tqdm
import os, glob
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import soundfile as sf
from dataset import make_dataset

from EaBNet import EaBNet, numParams, com_mag_mse_loss
from dataset.custom_dataset import CustomAudioVisualDataset
from dataset import make_dataset
from torch.utils.tensorboard import SummaryWriter


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]

def save_checkpoint(model, optimizer, iteration, filepath):
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        print('maybe useless cuz tensorboard create it ahead, but if not:')
        os.makedirs(folder)

    if isinstance(model, DDP):
        model = model.module 
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at '{filepath}'")

def load_checkpoint(model, optimizer, filepath):
    if not os.path.exists(filepath):
        print(f"Checkpoint '{filepath}' not found")
        return -1
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    print(f"Checkpoint loaded from '{filepath}', start from iteration {iteration}")
    
    return model, optimizer, iteration

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
 
def evaluate(is_master, model, device, criterion, valloader, iter, writer, args):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for i, (x, target) in enumerate(tqdm(valloader,desc=f'Valid:')):
            target = target.to(device)
            x = x.to(device)
            noisy_stft, target_stft = prepare_data(x, target, device, args)
            esti_stft = model(noisy_stft)
            #loss = criterion(esti_stft, target_stft)
            frame_list = [noisy_stft.shape[1]] * x.shape[0]
            loss = com_mag_mse_loss(esti_stft, target_stft, frame_list)
            torch.distributed.all_reduce(loss)
            loss_list.append(loss.item()/torch.distributed.get_world_size())
            
            #save an example
            if i == 0:
                sr = args.sr
                wav_len = int(args.wav_len * sr)
                win_size = int(args.win_size * sr)
                win_shift = int(args.win_shift * sr)
                fft_num = args.fft_num
                #esti_stft:[1, 2, 480, 161]  noisy_stft:[1, 480, 161, 4, 2]
                #stft ouput:[batch_size*mics, freq_num(161), seq_len, 2] [batch_size, freq_num, seq_len, 2]
                esti_stft, target_stft = esti_stft.permute(0, 3, 2, 1), target_stft.permute(0, 3, 2, 1)
                print(esti_stft.shape)
                esti_wav = torch.istft(torch.view_as_complex(esti_stft.contiguous()), fft_num, win_shift, win_size, torch.hann_window(win_size).to(device))
                esti_wav = esti_wav.cpu().numpy()   #[1, 76640]
                noisy_wav = x.squeeze(0).cpu().numpy()  #[4, 76672]
                target_wav = target.squeeze(0).cpu().numpy()    #[1, 76672]

                writer = writer or SummaryWriter(args.checkpoint_dir)
                writer.add_audio('estimated_audio', esti_wav, iter, args.sr)
                writer.add_audio('noisy_audio', noisy_wav[:1,:], iter, args.sr)
                writer.add_audio('target_audio', target_wav, iter, args.sr)

                #size of input tensor and input format are different.         tensor shape: (1, 161, 480, 2), input_format: CHW
                writer.add_image('estimated_spectrogram', esti_stft[..., 0], iter)
                writer.add_image('noisy_spectrogram', noisy_stft[..., 0, 0], iter)  #noisy形状不对 .transpose(1, 2)
                writer.add_image('target_spectrogram', target_stft[..., 0], iter)
                

    mean_loss = sum(loss_list)/len(loss_list)
    if is_master:    
        print('test_loss:', mean_loss)
        writer.add_scalar('valid_loss', mean_loss, iter)

    model.train()
    #return test_loss

def main(rank, world_size, port, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    is_master = (rank == 0)
    #writer = SummaryWriter(args.checkpoint_dir)
    writer = None
    # 初始化进程组
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    # 设置设备
    device = torch.device('cuda:{}'.format(rank))

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
                 ).to(device)
    
    #loss and optimizer
    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

    #resume from checkpoint
    current_iter = 0
    cplist = glob.glob(f"{args.checkpoint_dir}/*.pth")
    if len(cplist) > 0:
        paths = [int(os.path.basename(path).split('.')[0]) for path in cplist]
        checkpoint_path = cplist[paths.index(max(paths))]
        net, optimizer, resume_iter = load_checkpoint(net, optimizer, checkpoint_path)  
        if resume_iter != -1:
            current_iter = resume_iter + 1

    net = DDP(net, device_ids=[device])
    net.train()
    #print("The number of trainable parameters is:{}".format(numParams(net)))

    #dataset and dataloader
    tr_dataset, val_dataset = make_dataset(args)     #FIXME: 两个进程就超内存了wtf

    trainloader = utils.DataLoader(tr_dataset, args.batch_size, num_workers=args.num_workers, drop_last= True, sampler=DistributedSampler(tr_dataset, num_replicas=world_size, rank=rank))
    valloader = utils.DataLoader(val_dataset, 1, sampler=DistributedSampler(val_dataset, num_replicas=world_size, rank=rank))

    
    #-------------------------training loop-------------------------
    print('pid:', rank)
    #evaluate(is_master, net, device, loss, valloader, current_iter, writer, args)  #for debug
    
    #validation
    if is_master and args.validate_once_before_train:
        evaluate(is_master, net, device, loss, valloader, current_iter, writer, args)
        #write enhanced audio

    loss_list = []
    for epoch in range(args.total_epoch):
        #for i, x in enumerate(dataloader):
        for i, (x, target) in enumerate(tqdm(trainloader,desc=f'Epoch:{epoch}')):
            #临时cat4声道到9声道
            #x = torch.cat((x, x), dim=1)
            #x = torch.cat((x, x[:, 0:1, :]), dim=1) #[4, 9, 76672]
            
            optimizer.zero_grad()
            noisy_stft, target_stft = prepare_data(x, target, device, args)

            #[4, 480, 161, 4/9, 2]
            esti_stft = net(noisy_stft)

            #calculate loss
            frame_list = [noisy_stft.shape[1]] * args.batch_size
            l = com_mag_mse_loss(esti_stft, target_stft, frame_list)
            l.backward()
            optimizer.step()

            loss_list.append(l.item())
            if is_master:
                if current_iter % 50 == 0:
                    writer = writer or SummaryWriter(args.checkpoint_dir)   #lazy write
                    mean_loss = sum(loss_list)/len(loss_list)
                    writer.add_scalar('loss', mean_loss, current_iter)
                    loss_list = []
                
            current_iter += 1
        
        #-------------------------end of an epoch-------------------------
        print(f'end epoch {epoch}')
        #save checkpoint
        if is_master and (epoch % args.saving_interval == 0 or epoch == args.total_epoch - 1):
            writer.add_scalar('loss', mean_loss, current_iter)
            save_checkpoint(net, optimizer, current_iter, os.path.join(args.checkpoint_dir, f'{current_iter}.pth'))
        #validation
        if is_master and (epoch % args.valid_interval == 0 or epoch == args.total_epoch - 1):
            evaluate(is_master, net, device, loss, valloader, current_iter, writer, args)
            #write enhanced audio



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("This script provides the network code and a simple testing, you can train the"
                                     "network according to your own pipeline")
    #eabnet parameters
    parser.add_argument("--batch_size", type=int, default=6)    #8 in paper
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--total_epoch", type=int, default=200)  #60 in paper
    parser.add_argument("--mics", type=int, default=4)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--wav_len", type=float, default=6.0)
    parser.add_argument("--win_size", type=float, default=0.020)
    parser.add_argument("--win_shift", type=float, default=0.010)
    parser.add_argument("--fft_num", type=int, default=320)
    parser.add_argument("--k1", type=tuple, default=(2,3))
    parser.add_argument("--k2", type=tuple, default=(1,3))
    parser.add_argument("--c", type=int, default=64)
    parser.add_argument("--M", type=int, default=4)
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
    parser.add_argument("--valid_interval", type=int, default=4)

    #dataset parameters     processed是4声道，processed1是8声道，但是加载时超内存
    processed_folder = 'processed'
    parser.add_argument('--training_predictors_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_target_test.pkl')
    parser.add_argument('--dataset', type=str, default='l3das23', choices=['l3das23', 'mcse'])

    #saving parameters
    from datetime import datetime
    exptime = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    #parser.add_argument('--expname', type=str, default='',
    #                    help='Experiment name')
    parser.add_argument('--results_path', type=str, default=f'/data/wbh/l3das23/experiment/{exptime}/results',
                        help='Folder to write results dicts into')
    parser.add_argument('--checkpoint_dir', type=str, default=f'/data/wbh/l3das23/experiment/{exptime}/checkpoints',
                        help='Folder to write checkpoints into')
    parser.add_argument('--path_images', type=str, default=None,
                        help="Path to the folder containing all images of Task1. None when using the audio-only version")
    parser.add_argument('--path_csv_images_train', type=str, default='/data/wbh/l3das23/L3DAS23_Task1_train/audio_image.csv',
                        help="Path to the CSV file for the couples (name_audio, name_photo) in the train/val set")
    parser.add_argument('--path_csv_images_test', type=str, default='/data/wbh/l3das23/L3DAS23_Task1_dev/audio_image.csv',
                        help="Path to the CSV file for the couples (name_audio, name_photo)")
    parser.add_argument("--saving_interval", type=int, default=1)
    parser.add_argument('--validate_once_before_train', action='store_true', default=False)

    args = parser.parse_args()

    #specify the path to load checkpoints
    #args.checkpoint_dir = '/data/wbh/l3das23/experiment/2023-11-23-14:22:43/checkpoints/'
    #args.results_path = '/data/wbh/l3das23/experiment/2023-11-23-14:22:43/results/'
    # args.checkpoint_dir = '/data/wbh/l3das23/experiment/debug/checkpoints/'
    # args.results_path = '/data/wbh/l3das23/experiment/debug/results/'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    world_size = torch.cuda.device_count()
    port = _get_free_port()
    torch.multiprocessing.spawn(main, args=(world_size, port, args, ), nprocs=world_size)
    #main(args)