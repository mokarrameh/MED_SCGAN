"""Train SPA GAN, the authors' re-implementation (comparison model in Tables 4 and 6).

This is not the official SPA GAN code; see models/spa_gan.py.
Generators: ResNet with 9 residual blocks and a spatial-attention head; PatchGAN
discriminators. Losses (weights in models/spa_gan.SpaGanLossWeights): least-squares GAN,
cycle-consistency L1 x 10, identity L1 x 0.5, attention sparsity x 1, attention total
variation x 1.
Preprocessing differs from the other models: resize (shorter side 286, bicubic) and
horizontal flip only, without random crop or rotation. The learning rate is fixed at 2e-4
in model.setup_optim(); --lr is not used. torch.manual_seed(0) is set. Requires a CUDA GPU.
Checkpoints are written to checkpoints/spa_res9/ after every epoch.
Losses and images are sent to a Visdom server (VISDOM_SERVER, default http://localhost:8097).
"""
import argparse

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.utils import LambdaLR
from PIL import Image
import torch

from models.spa_gan import *

from utils.utils import Logger
from utils.datasets import ImageDataset

import os

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

save_dir = 'checkpoints/spa_res9'

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

torch.manual_seed(0)

# Networks
model = SpaGanModel(img_nc=3, ngf=64, ndf=64, n_res_blocks=9, device="cuda")

model.setup_optim(lr=2e-4)
w = SpaGanLossWeights()

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(model.opt_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(model.opt_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader), env='spa_res9')
os.makedirs(save_dir, exist_ok=True)
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    logger.start_epoch_timer()
    for i, batch in enumerate(dataloader):
        a = batch['A'].cuda()
        b = batch['B'].cuda()

        batch = SpaGanBatch(a, b)

        logs = model.training_step(batch, w)

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': logs['G_total'], 'loss_gan': logs['gan'],'Loss_Attn': (logs['attn_tv']), 'loss_identity': logs['identity'], 'attn_sparsity': logs['attn_sparsity'],'cycle': logs['cycle'], 'loss_D': logs['D_total']}, images={'real_A': logs['real_a'], 'real_B': logs['real_b'], 'fake_A': logs['fake_a'], 'fake_B': logs['fake_b']})

    old_lr = model.opt_G.param_groups[0]['lr']
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()

    lr_G = model.opt_G.param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr_G))
    # Save models checkpoints
    torch.save(model.G_AB.state_dict(), 'checkpoints/spa_res9/netG_A2B.pth')
    torch.save(model.G_BA.state_dict(), 'checkpoints/spa_res9/netG_B2A.pth')
    torch.save(model.D_A.state_dict(), 'checkpoints/spa_res9/netD_A.pth')
    torch.save(model.D_B.state_dict(), 'checkpoints/spa_res9/netD_B.pth')
    

# Compute average epoch time
avg_epoch_time = sum(logger.epoch_times) / len(logger.epoch_times)
print(f"Average training time per epoch: {avg_epoch_time:.2f} sec")

    
