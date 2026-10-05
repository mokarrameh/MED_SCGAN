#!/bin/bash
# Slurm job for Table 3, experiment 4 (Style-Enhanced CycleGAN, U-Net) on the USask Plato cluster.
#SBATCH --job-name=scgan_unet          # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=3g.40gb:1   # an A100 MIG slice, not a full A100 (see README)
#SBATCH --mem=32GB                   # Memory required
#SBATCH --time=4-00:00:00              # Time limit, adjust as needed
#SBATCH --output=scgan_unet_output.txt # Output file
#SBATCH --account=<your-account>     # Slurm allocation

# Modules on the USask Plato cluster (CUDA 12.2, Python 3.11, SciPy stack);
# replace them with the equivalents on your system.
module load cuda/12.2                # Load CUDA (adjust version)
module load python/3.11               # Load Python
module load scipy-stack

# Activate virtual environment
source <path-to-venv>/bin/activate     # virtual environment with requirements.txt installed

cd <path-to-MED_SCGAN>
# export VISDOM_SERVER=http://<host>   # optional: Visdom server for live loss plots

# Train the model
python3 style_train_unet.py --dataroot datasets/scgan_comp/ --cuda

# Test the model
python3 style_test_unet.py --dataroot datasets/scgan_comp/ --cuda

python utils/evaluate_models.py --gt "datasets/scgan_comp/test/A" --models SCGAN_UNET="checkpoints/scgan_unet/A" --kid_subset_size 100

python utils/evaluate_models.py --gt "datasets/scgan_comp/test/B" --models SCGAN_UNET="checkpoints/scgan_unet/B" --kid_subset_size 100

# Deactivate the virtual environment
deactivate