"""Inference time per image of the CUT generator (Table 6); runs inside a CUT checkout.

CUT is not part of this repository. Copy this file into a clone of
https://github.com/taesungp/contrastive-unpaired-translation (it imports CUT's
options.test_options and models), put the trained generator at
checkpoints/cut/latest_net_G.pth, set img_path below to a test image, and run it from the
root of that clone. Command-line options are parsed by CUT's TestOptions.
Same protocol as measure_inference_time.py: 10 warm-up passes, then 100 timed passes.
"""
import torch
import time
from PIL import Image
from torchvision import transforms
from options.test_options import TestOptions
from models import create_model

# -----------------------------
# Settings
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = "datasets/scgan_comp/test/A/003_VP3_frame0030.png"  # replace with any test image
img_size = 256
num_warmup = 10
num_runs = 100

# -----------------------------
# Prepare image
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# -----------------------------
# Define which CUT models to benchmark
# -----------------------------
models_to_test = {
    "cut": "checkpoints/cut/latest_net_G.pth"
}

# -----------------------------
# Measure inference time
# -----------------------------
for name, ckpt_path in models_to_test.items():
    print(f"\nEvaluating {name} ...")

    # Setup test options (minimal)
    opt = TestOptions().parse()
    opt.model = "cut"
    opt.name = name
    opt.phase = "test"
    opt.no_dropout = True
    opt.isTrain = False

    # Create CUT model and load weights
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    model.netG.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.netG.to(device).eval()

    # Warm-up
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.netG(input_tensor)

    # Measure
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            starter.record()
            _ = model.netG(input_tensor)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))  # ms

    mean_time = sum(timings) / len(timings)
    std_time = (sum((t - mean_time) ** 2 for t in timings) / len(timings)) ** 0.5
    print(f"{name} inference time: {mean_time:.2f} ± {std_time:.2f} ms per image")

torch.cuda.empty_cache()
