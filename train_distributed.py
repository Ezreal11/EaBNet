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

from EaBNet import EaBNet, numParams, com_mag_mse_loss
from dataset.custom_dataset import CustomAudioVisualDataset
from torch.utils.tensorboard import SummaryWriter

def load_dataset(args):
    #LOAD DATASET
    print ('Loading dataset')

    with open(args.training_predictors_path, 'rb') as f:
        training_audio_predictors = pickle.load(f)
    with open(args.training_target_path, 'rb') as f:
        training_target = pickle.load(f)
    '''with open(args.validation_predictors_path, 'rb') as f:
        validation_audio_predictors = pickle.load(f)
    with open(args.validation_target_path, 'rb') as f:
        validation_target = pickle.load(f)
    with open(args.test_predictors_path, 'rb') as f:
        test_predictors = pickle.load(f)
    with open(args.test_target_path, 'rb') as f:
        test_target = pickle.load(f)'''
    
    training_img_predictors = training_audio_predictors[1]
    training_audio_predictors = np.array(training_audio_predictors[0])
    training_target = np.array(training_target)
    #validation_img_predictors = validation_audio_predictors[1]
    #validation_audio_predictors = np.array(validation_audio_predictors[0])
    # validation_img_predictors = validation_predictors[1]
    #validation_target = np.array(validation_target)
    #test_audio_predictors = np.array(test_predictors[0])
    #test_img_predictors = test_predictors[1]
    #test_target = np.array(test_target)

    print ('\nShapes:')
    print ('Training predictors: ', training_audio_predictors.shape)
    #print ('Validation predictors: ', validation_audio_predictors.shape)
    #print ('Test predictors: ', test_audio_predictors.shape)

    #convert to tensor
    training_audio_predictors = torch.tensor(training_audio_predictors).float()
    #validation_audio_predictors = torch.tensor(validation_audio_predictors).float()
    #test_audio_predictors = torch.tensor(test_audio_predictors).float()
    training_target = torch.tensor(training_target).float()
    #validation_target = torch.tensor(validation_target).float()
    #test_target = torch.tensor(test_target).float()
    
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
    #tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
    #val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
    #test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)
    return tr_dataset  #, val_data, test_data

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
    return iteration

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
        resume_iter = load_checkpoint(net, optimizer, cplist[-1])
        if resume_iter != -1:
            current_iter = resume_iter + 1

    net = DDP(net, device_ids=[device])
    net.train()
    print("The number of trainable parameters is:{}".format(numParams(net)))


    batch_size = args.batch_size
    mics = args.mics
    sr = args.sr
    wav_len = int(args.wav_len * sr)
    win_size = int(args.win_size * sr)
    win_shift = int(args.win_shift * sr)
    fft_num = args.fft_num

    #dataset and dataloader
    tr_dataset = load_dataset(args)     #FIXME: 两个进程就超内存了wtf
    dataloader = utils.DataLoader(tr_dataset, args.batch_size, sampler=DistributedSampler(tr_dataset, num_replicas=world_size, rank=rank))
    
    loss_list = []
    #training loop
    print('pid:', rank)
    for epoch in range(args.total_epoch):
        #for i, x in enumerate(dataloader):
        for i, (x, target) in enumerate(tqdm(dataloader)):
            #current_iter = epoch * len(dataloader) + i
            #print('pid:',rank,"x:",x)
            #x:[4, 4, 76672]
            #FIXME:临时cat4声道到9声道
            x = torch.cat((x, x), dim=1)
            x = torch.cat((x, x[:, 0:1, :]), dim=1) #[4, 9, 76672]
            optimizer.zero_grad()
            noisy_wav = x.to(device)    #[4, 4, 76672]
            #target_wav = torch.rand(args.batch_size, wav_len).cuda()
            target_wav = target.to(device)  #[4, 1, 76672]
            noisy_wav = noisy_wav.contiguous().view(batch_size*mics, -1)#noisy_wav.shape[-1]) #[batch_size*mics, wav_len]
            target_wav = target.squeeze(1)
            noisy_stft = torch.stft(noisy_wav, fft_num, win_shift, win_size, torch.hann_window(win_size).to(noisy_wav.device))
            target_stft = torch.stft(target_wav, fft_num, win_shift, win_size, torch.hann_window(win_size).to(target_wav.device))
            _, freq_num, seq_len, _ = noisy_stft.shape
            noisy_stft = noisy_stft.view(batch_size, mics, freq_num, seq_len, -1).permute(0, 3, 2, 1, 4).to(device)
            target_stft = target_stft.permute(0, 3, 2, 1).to(device)
            # conduct sqrt power-compression
            noisy_mag, noisy_phase = torch.norm(noisy_stft, dim=-1) ** 0.5, torch.atan2(noisy_stft[..., -1], noisy_stft[..., 0])
            target_mag, target_phase = torch.norm(target_stft, dim=1) ** 0.5, torch.atan2(target_stft[:, -1, ...], target_stft[:, 0, ...])
            noisy_stft = torch.stack((noisy_mag * torch.cos(noisy_phase), noisy_mag * torch.sin(noisy_phase)), dim=-1).to(device)
            target_stft = torch.stack((target_mag * torch.cos(target_phase), target_mag * torch.sin(target_phase)), dim=1).to(device)

            #[4, 480, 161, 4/9, 2]
            esti_stft = net(noisy_stft)
            #esti_stft = target_stft

            #calculate loss
            #l = loss(esti_stft, target_stft)
            frame_list = [noisy_stft.shape[1]] * args.batch_size
            #frame_list = [(wav_len - win_size + win_size) // win_shift + 1]*args.batch_size
            
            l = com_mag_mse_loss(esti_stft, target_stft, frame_list)
            #l = loss(esti_stft, target_stft)
            #print('loss:', l.item())

            l.backward()
            optimizer.step()

            loss_list.append(l.item())

            if is_master:
                if current_iter % 50 == 0:
                    writer = writer or SummaryWriter(args.checkpoint_dir)   #lazy write
                    mean_loss = sum(loss_list)/len(loss_list)
                    print('iter:', current_iter, 'mean_loss:', mean_loss)
                    writer.add_scalar('loss', mean_loss, current_iter)
                    loss_list = []
                if current_iter + 1 % len(dataloader) == 0:
                    save_checkpoint(net, optimizer, current_iter, os.path.join(args.checkpoint_dir, f'{current_iter}.pth'))

            current_iter += 1
        
        #end of an epoch
        print(f'end epoch {epoch}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("This script provides the network code and a simple testing, you can train the"
                                     "network according to your own pipeline")
    #eabnet parameters
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

    #dataset parameters     processed是4声道，processed1是8声道，但是加载时超内存
    processed_folder = 'processed'
    parser.add_argument('--training_predictors_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default=f'/data/wbh/l3das23/{processed_folder}/task1_target_test.pkl')
    
    #saving parameters
    from datetime import datetime
    exptime = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
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

    args = parser.parse_args()

    #specify the path to load checkpoints
    #args.checkpoint_dir = '/data/wbh/l3das23/experiment/2023-11-21-17:11:56/checkpoints/'
    #args.results_path = '/data/wbh/l3das23/experiment/2023-11-21-17:11:56/results/'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    world_size = torch.cuda.device_count()
    port = _get_free_port()
    torch.multiprocessing.spawn(main, args=(world_size, port, args, ), nprocs=world_size)
    #main(args)