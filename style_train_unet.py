"""Train the Style-Enhanced CycleGAN with a 7-level U-Net generator (Table 3, experiment 4).

Data: <dataroot>/train/A (WLI) and <dataroot>/train/B (NBI); see prepare_data.py.
Generators: models/style_models.UnetGenerator (num_downs=7, mapping network of 8 fully
connected layers, z and w of dimension 512, AdaIN after the convolutions); PatchGAN
discriminators. Losses: least-squares GAN (MSE), cycle-consistency L1 x 10, identity L1 x 5,
plus a VGG-19 Gram-matrix style loss weighted by --lambda_style (default 0.01). The style
loss is not described in the paper (see README).
UnetGenerator.forward defaults to use_w_avg=True, so every generator call here uses the
running-average style vector w_avg and the mapping network receives no gradient (see README).
Checkpoints are written to checkpoints/scgan_unet/ after every epoch.
Losses and images are sent to a Visdom server (VISDOM_SERVER, default http://localhost:8097).
Adapted from PyTorch-CycleGAN (https://github.com/ai-tor/PyTorch-CycleGAN, GPL-3.0).
"""
import argparse
import itertools
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn

from models.style_models import VGGStyleLoss

from models.style_models import UnetGenerator, define_D, init_net

from utils.utils import ReplayBuffer
from utils.utils import LambdaLR
from utils.utils import Logger, weights_init_normal
from utils.datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--lambda_style', type=float, default=0.01, help='weight for style loss')
opt = parser.parse_args()
print(opt)

# Set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Latent Dimension
z_dim = 512
w_dim = 512  # Latent vector dimension for StyleGAN features

netG_A2B = init_net(UnetGenerator(input_nc=3, output_nc=3, num_downs=7, z_dim=z_dim, w_dim=w_dim, ngf=64).to(device))
netG_B2A = init_net(UnetGenerator(input_nc=3, output_nc=3, num_downs=7, z_dim=z_dim, w_dim=w_dim, ngf=64).to(device))

# Discriminators remain the same
netD_A = define_D(opt.output_nc, 64, 'basic', 3, 'instance', 'normal', 0.02).to(device)
netD_B = define_D(opt.input_nc, 64, 'basic', 3, 'instance', 'normal', 0.02).to(device)

# netG_A2B.apply(weights_init_normal)
# netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Losses
criterion_GAN = torch.nn.MSELoss()                       # GAN loss (LSGAN for stability)
criterion_cycle = nn.L1Loss()                            # Cycle-consistency loss
criterion_identity = nn.L1Loss()                         # Identity loss for same-domain mapping
criterion_style = VGGStyleLoss(device=device)         # Optional: Style loss for latent mapping

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
    lr=opt.lr, betas=(0.5, 0.999)
)
optimizer_D_A = torch.optim.Adam(
    netD_A.parameters(),
    lr=opt.lr, betas=(0.5, 0.999)
)

optimizer_D_B = torch.optim.Adam(
    netD_B.parameters(),
    lr=opt.lr, betas=(0.5, 0.999)
)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = torch.ones(opt.batchSize, device=device, requires_grad=False)
target_fake = torch.zeros(opt.batchSize, device=device, requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.LANCZOS), 
                transforms.RandomCrop(opt.size),
                #transforms.RandomAffine(degrees=45,translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                #transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader), env='scgan_unet')

# Checkpoint folder (torch.save below does not create it)
os.makedirs('checkpoints/scgan_unet', exist_ok=True)
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    logger.start_epoch_timer()
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Sample z for style injection
        z_A = torch.randn(real_A.size(0), z_dim, device=device)
        z_B = torch.randn(real_B.size(0), z_dim, device=device)
        #z_A = torch.normal(mean=0, std=0.2, size=(real_A.size(0), z_dim)).cuda()
        #z_B = torch.normal(mean=0, std=0.2, size=(real_B.size(0), z_dim)).cuda()

        #### ---------------------------
        ####  Train Generators
        #### ---------------------------
        optimizer_G.zero_grad()

        # Identity
        idt_A = netG_B2A(real_A, z_A, use_w_avg=True)
        loss_idt_A = criterion_identity(idt_A, real_A) * 5.0

        idt_B = netG_A2B(real_B, z_B, use_w_avg=True)
        loss_idt_B = criterion_identity(idt_B, real_B) * 5.0

        # Forward cycle
        fake_B = netG_A2B(real_A, z_A)
        pred_fake = netD_B(fake_B)
        loss_G_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B, z_B)
        pred_fake = netD_A(fake_A)
        loss_G_B2A = criterion_GAN(pred_fake, target_real)
        
        # Cycle losses
        rec_A = netG_B2A(fake_B, z_A)
        loss_cycle_A = criterion_cycle(rec_A, real_A) * 10.0

        rec_B = netG_A2B(fake_A, z_B)
        loss_cycle_B = criterion_cycle(rec_B, real_B) * 10.0

        # Optional: Style loss
        loss_style_A = criterion_style(fake_A, real_A) * opt.lambda_style # Generated A should have style of real A
        loss_style_B = criterion_style(fake_B, real_B) * opt.lambda_style # Generated B should have style of real B

        # Total generator loss
        loss_G = (
            loss_G_A2B + 
            loss_G_B2A +
            loss_cycle_A +
            loss_cycle_B +
            loss_idt_A + 
            loss_idt_B + 
            loss_style_A + 
            loss_style_B
        )

        loss_G.backward()
        optimizer_G.step()

        #### ---------------------------
        ####  Train Discriminators
        #### ----------------------
        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_A_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_A_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()
        # Real loss
        pred_real = netD_B(real_B)
        loss_D_B_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_B_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        ###### Logging ######
        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G,
                    'loss_G_identity': (loss_idt_A + loss_idt_B),
                    'loss_G_GAN': (loss_G_A2B + loss_G_B2A),
                    'loss_G_cycle': (loss_cycle_A + loss_cycle_B),
                    'loss_D': (loss_D_A + loss_D_B),
                    'loss_style': (loss_style_A + loss_style_B)}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    old_lr = optimizer_G.param_groups[0]['lr']
    ###### Update Learning Rates ######
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    lr_G = optimizer_G.param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr_G))

    ###### Save Models ######
    torch.save(netG_A2B.state_dict(), "checkpoints/scgan_unet/netG_A2B.pth")
    torch.save(netG_B2A.state_dict(), "checkpoints/scgan_unet/netG_B2A.pth")
    torch.save(netD_A.state_dict(), "checkpoints/scgan_unet/netD_A.pth")
    torch.save(netD_B.state_dict(), "checkpoints/scgan_unet/netD_B.pth")

###################################
# Compute average epoch time
avg_epoch_time = sum(logger.epoch_times) / len(logger.epoch_times)
print(f"Average training time per epoch: {avg_epoch_time:.2f} sec")