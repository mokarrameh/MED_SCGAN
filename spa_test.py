"""Translate the test images with SPA GAN, the authors' re-implementation (9 residual blocks).

Reads <dataroot>/test/A and <dataroot>/test/B, resizes every image to 256 x 256 (bicubic)
and writes the translations to checkpoints/spa_res9/B (WLI -> NBI) and
checkpoints/spa_res9/A (NBI -> WLI), keeping the input file names.
The paper's FID and KID compare test/B with the B folder (see utils/evaluate_models.py).
Requires a CUDA GPU.
"""
import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image
import torch

from models.spa_gan import AttnResnetGenerator
from utils.datasets import ImageDatasetPath

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='checkpoints/spa_res9/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='checkpoints/spa_res9/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = AttnResnetGenerator(opt.input_nc, opt.output_nc, ngf=64, n_blocks=9)
netG_B2A = AttnResnetGenerator(opt.input_nc, opt.output_nc, ngf=64, n_blocks=9)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B, weights_only=True))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A, weights_only=True))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

transforms_ = [transforms.Resize((256, 256), Image.BICUBIC), 
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDatasetPath(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('checkpoints/spa_res9/A'):
    os.makedirs('checkpoints/spa_res9/A')
if not os.path.exists('checkpoints/spa_res9/B'):
    os.makedirs('checkpoints/spa_res9/B')

for i, batch in enumerate(dataloader):
    # Set model input
    a = batch['A'].cuda()
    b = batch['B'].cuda()

    # Generate output
    fake_B, _, _ = netG_A2B(a) 
    fake_B = 0.5*(fake_B + 1.0)
    fake_A, _, _ = netG_B2A(b)
    fake_A = 0.5*(fake_A + 1.0)

    # Extract real image filenames
    real_A_name = os.path.basename(batch['A_path'][0])  # Assuming `batch['A_path']` has file paths for real_A
    real_B_name = os.path.basename(batch['B_path'][0])  # Assuming `batch['B_path']` has file paths for real_B

    # Save image files
    save_image(fake_A, f'checkpoints/spa_res9/A/{real_B_name}')
    save_image(fake_B, f'checkpoints/spa_res9/B/{real_A_name}')

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
