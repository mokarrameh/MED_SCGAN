"""Inference time per image of the WLI -> NBI generators on a CUDA GPU (Table 6).

For each generator: 10 warm-up passes, then 100 timed passes on one 256 x 256 image
(batch size 1) measured with CUDA events; prints mean and standard deviation in ms.
The SPA GAN generator is built with 6 residual blocks and loaded from checkpoints/spa_res6/,
while spa_train.py and spa_test.py use 9 blocks (see README).
"""
import argparse
import torch
import time
from torchvision import transforms
from PIL import Image
from models.cyclegan_models import define_G
from models.style_models import ResNetGenerator, UnetGenerator
from models.spa_gan import AttnResnetGenerator

def infer(model_path, input_tensor, name, netG):
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.to(device).eval()

    # Warm-up runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = netG(input_tensor)
    torch.cuda.synchronize()

    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = netG(input_tensor)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))  # ms

    mean_time = sum(timings) / len(timings)
    std_time = (sum((t - mean_time) ** 2 for t in timings) / len(timings)) ** 0.5

    print(f"{name} average inference time: {mean_time:.2f} ± {std_time:.2f} ms per image")

def infer_z(model_path, input_tensor, name, netG, z):
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.to(device).eval()

    # Warm-up runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = netG(input_tensor, z)
    torch.cuda.synchronize()

    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = netG(input_tensor, z)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))  # ms

    mean_time = sum(timings) / len(timings)
    std_time = (sum((t - mean_time) ** 2 for t in timings) / len(timings)) ** 0.5

    print(f"{name} average inference time: {mean_time:.2f} ± {std_time:.2f} ms per image")

# -----------------------------
# Settings
# -----------------------------
parser = argparse.ArgumentParser(description="Inference time of the WLI->NBI generators on a CUDA GPU (Table 6).")
parser.add_argument('--image', default="datasets/scgan_comp/test/A/003_VP3_frame0030.png", help='any test image')
parser.add_argument('--cycle_res9', default="checkpoints/cycle_res9_comp/netG_A2B.pth", help='CycleGAN, ResNet-9 generator')
parser.add_argument('--cycle_unet', default="checkpoints/cycle_unet/netG_A2B.pth", help='CycleGAN, U-Net generator')
parser.add_argument('--style_res9', default="checkpoints/scgan_res9/netG_A2B.pth", help='Style-Enhanced CycleGAN, ResNet-9 generator')
parser.add_argument('--style_unet', default="checkpoints/scgan_unet/netG_A2B.pth", help='Style-Enhanced CycleGAN, U-Net generator')
parser.add_argument('--spa', default="checkpoints/spa_res6/netG_A2B.pth", help="SPA GAN, the authors' re-implementation (built with n_blocks=6 below)")
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = opt.image  # any test image
img_size = 256
input_nc = 3
output_nc = 3
num_warmup = 10
num_runs = 100

# -----------------------------
# Define image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# -----------------------------
# List models to compare
# -----------------------------
models = {
    "CycleGAN_Res9": opt.cycle_res9,
    "CycleGAN_UNet": opt.cycle_unet,
    "StyleEnhanced_Res9": opt.style_res9,
    "StyleEnhanced_UNet": opt.style_unet,
    "SPA_GAN": opt.spa
}

# -----------------------------
# Benchmark loop
# -----------------------------
print(f"\nEvaluating CycleGAN_Res9 ...")
netG_A2B = define_G(input_nc=input_nc, output_nc=output_nc, ngf=64, netG='resnet_9blocks', norm='instance', use_dropout=True)
infer(models["CycleGAN_Res9"], input_tensor, "CycleGAN_Res9", netG_A2B)
torch.cuda.empty_cache()

print(f"\nEvaluating CycleGAN_UNet ...")
netG_A2B = define_G(input_nc=input_nc, output_nc=output_nc, ngf=64, netG='unet_128', norm='instance', use_dropout=True)
infer(models["CycleGAN_UNet"], input_tensor, "CycleGAN_UNet", netG_A2B)
torch.cuda.empty_cache()


print(f"\nEvaluating StyleEnhanced_Res9 ...")
z = torch.randn(input_tensor.shape[0], 512, device=device)
netG_A2B = ResNetGenerator(z_dim=512, w_dim=512, img_channels=input_nc, n_blocks=9)
infer_z(models["StyleEnhanced_Res9"], input_tensor, "StyleEnhanced_Res9", netG_A2B, z)
torch.cuda.empty_cache()


print(f"\nEvaluating StyleEnhanced_UNet ...")
z = torch.randn(input_tensor.shape[0], 512, device=device)
netG_A2B = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=7, z_dim=512, w_dim=512, ngf=64)
infer_z(models["StyleEnhanced_UNet"], input_tensor, "StyleEnhanced_UNet", netG_A2B, z)
torch.cuda.empty_cache()

print(f"\nEvaluating SPA_GAN ...")
# Note: 6 residual blocks here; spa_train.py and spa_test.py use 9 (see README).
netG_A2B = AttnResnetGenerator(input_nc, output_nc, ngf=64, n_blocks=6)
infer(models["SPA_GAN"], input_tensor, "SPA_GAN", netG_A2B)

torch.cuda.empty_cache()
