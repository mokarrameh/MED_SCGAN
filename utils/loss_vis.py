"""Plot per-epoch loss curves (Fig. 9) from the log of a *_train.py run.

Usage: python utils/loss_vis.py --log_file scgan_unet_output.txt --out loss_curves.svg
"""
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Plot per-epoch loss curves from a training log.")
parser.add_argument('--log_file', required=True, help='stdout of a *_train.py run (e.g. the Slurm output file)')
parser.add_argument('--out', default='loss_curves.svg', help='output figure (SVG)')
args = parser.parse_args()

# Path to your log file
log_file = args.log_file

# Regular expression to extract epoch and loss values
pattern = re.compile(r"Epoch (\d+)/\d+ \[\d+/\d+\] -- loss_G: ([\d\.]+) \| loss_G_identity: ([\d\.]+) \| loss_G_GAN: ([\d\.]+) \| loss_G_cycle: ([\d\.]+) \| loss_D: ([\d\.]+)")

# Dictionary to store loss sums and counts per epoch
epoch_losses = {}

# Read and parse log file
with open(log_file, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            losses = list(map(float, match.groups()[1:]))

            if epoch not in epoch_losses:
                epoch_losses[epoch] = {"sum": np.array(losses), "count": 1}
            else:
                epoch_losses[epoch]["sum"] += np.array(losses)
                epoch_losses[epoch]["count"] += 1

# Compute average loss per epoch
epochs = sorted(epoch_losses.keys())
avg_loss_G = []
avg_loss_G_identity = []
avg_loss_G_GAN = []
avg_loss_G_cycle = []
avg_loss_D = []

for epoch in epochs:
    avg_losses = epoch_losses[epoch]["sum"] / epoch_losses[epoch]["count"]
    avg_loss_G.append(avg_losses[0])
    avg_loss_G_identity.append(avg_losses[1])
    avg_loss_G_GAN.append(avg_losses[2])
    avg_loss_G_cycle.append(avg_losses[3])
    avg_loss_D.append(avg_losses[4])

# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(epochs, avg_loss_G, label="Total Loss", color='blue')
plt.plot(epochs, avg_loss_G_identity, label="Identity Loss", color='orange')
plt.plot(epochs, avg_loss_G_GAN, label="Generator Loss", color='green')
plt.plot(epochs, avg_loss_G_cycle, label="Cycle Loss", color='red')
plt.plot(epochs, avg_loss_D, label="Descriminator Loss", color='purple')

plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("Loss Curves")
plt.legend()
plt.grid()

# Save the figure to file
plt.savefig(args.out, format="svg", bbox_inches="tight")

plt.show()