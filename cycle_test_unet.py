"""Translate the test images with the baseline CycleGAN, U-Net generator (experiment 2).

Reads <dataroot>/test/A and <dataroot>/test/B, resizes every image to 256 x 256 (bicubic)
and writes the translations to checkpoints/cycle_unet/B (WLI -> NBI) and
checkpoints/cycle_unet/A (NBI -> WLI), keeping the input file names.
The paper's FID and KID compare test/B with the B folder (see utils/evaluate_models.py).
"""
import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models.cyclegan_models import define_G
from utils.datasets import ImageDatasetPath

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='checkpoints/cycle_unet/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='checkpoints/cycle_unet/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=64, netG='unet_128', norm='instance', use_dropout=True)
netG_B2A = define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=64, netG='unet_128', norm='instance', use_dropout=True)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B, weights_only=True))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A, weights_only=True))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

transforms_ = [transforms.Resize((256, 256), Image.BICUBIC), 
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDatasetPath(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('checkpoints/cycle_unet/A'):
    os.makedirs('checkpoints/cycle_unet/A')
if not os.path.exists('checkpoints/cycle_unet/B'):
    os.makedirs('checkpoints/cycle_unet/B')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Extract real image filenames
    real_A_name = os.path.basename(batch['A_path'][0])  # Assuming `batch['A_path']` has file paths for real_A
    real_B_name = os.path.basename(batch['B_path'][0])  # Assuming `batch['B_path']` has file paths for real_B

    # Save image files
    save_image(fake_A, f'checkpoints/cycle_unet/A/{real_B_name}')
    save_image(fake_B, f'checkpoints/cycle_unet/B/{real_A_name}')

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
