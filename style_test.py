"""Translate the test images with the Style-Enhanced CycleGAN, ResNet-9 generator (experiment 3).

Reads <dataroot>/test/A and <dataroot>/test/B, resizes every image to 256 x 256 (bicubic)
and writes the translations to <checkpoint_dir>/B (WLI -> NBI) and
<checkpoint_dir>/A (NBI -> WLI), keeping the input file names.
The paper's FID and KID compare test/B with the B folder (see utils/evaluate_models.py).
A new latent vector z ~ N(0, I) is drawn for every image and no random seed is set, so
repeated runs give different images. --latent_dim is parsed but not used (z has
dimension 512).
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
import torch.nn as nn

from models.style_models import ResNetGenerator  # Import latent mapping network
from utils.datasets import ImageDatasetPath

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--latent_dim', type=int, default=512, help='dimension of the latent space z')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/scgan_res9', help='path to the checkpoint directory')
parser.add_argument('--rgb_conversion', action='store_true', help='Use this flag if you have grayscale images')

opt = parser.parse_args()
print(opt)

# Set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Latent Dimension
z_dim = 512
w_dim = 512  # Latent vector dimension for StyleGAN features

# Networks
netG_A2B = ResNetGenerator(z_dim=z_dim, w_dim=w_dim, img_channels=opt.input_nc, n_blocks=9)
netG_B2A = ResNetGenerator(z_dim=z_dim, w_dim=w_dim, img_channels=opt.input_nc, n_blocks=9)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(f'{opt.checkpoint_dir}/netG_A2B.pth', weights_only=True))
netG_B2A.load_state_dict(torch.load(f'{opt.checkpoint_dir}/netG_B2A.pth', weights_only=True))

# Set models to evaluation mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
if opt.rgb_conversion:
    transforms_ = [transforms.Resize((256, 256), Image.BICUBIC), 
                   transforms.Lambda(lambda img: img.convert("RGB")),   # <--- ensure RGB
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
else:
    transforms_ = [transforms.Resize((256, 256), Image.BICUBIC), 
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDatasetPath(opt.dataroot, transforms_=transforms_, mode='test'),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing ######

# Create output dirs if they don't exist
if not os.path.exists(f'{opt.checkpoint_dir}/A'):
    os.makedirs(f'{opt.checkpoint_dir}/A')
if not os.path.exists(f'{opt.checkpoint_dir}/B'):
    os.makedirs(f'{opt.checkpoint_dir}/B')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Sample z for style injection
    z_A = torch.randn(real_A.size(0), z_dim, device=device)
    z_B = torch.randn(real_B.size(0), z_dim, device=device)

    # Generate output with style modulation
    fake_B = 0.5 * (netG_A2B(real_A, z_A).data + 1.0)
    fake_A = 0.5 * (netG_B2A(real_B, z_B).data + 1.0)

    # Extract real image filenames
    real_A_name = os.path.basename(batch['A_path'][0])  # Assuming `batch['A_path']` has file paths for real_A
    real_B_name = os.path.basename(batch['B_path'][0])  # Assuming `batch['B_path']` has file paths for real_B

    # Save image files
    save_image(fake_A, f'{opt.checkpoint_dir}/A/{real_B_name}')
    save_image(fake_B, f'{opt.checkpoint_dir}/B/{real_A_name}')

    sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

sys.stdout.write('\n')
###################################
