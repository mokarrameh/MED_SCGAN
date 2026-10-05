#!/bin/bash
# Slurm job for SPA GAN (the authors' re-implementation) on the USask Plato cluster.
#SBATCH --job-name=spa_res9          # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32GB                   # Memory required
#SBATCH --time=2-00:00:00              # Time limit, adjust as needed
#SBATCH --output=spa_res9_output.txt # Output file
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
python3 spa_train.py --dataroot datasets/scgan_complete/ --cuda

# Test the model
python3 spa_test.py --dataroot datasets/scgan_complete/ --cuda --generator_A2B "checkpoints/spa_res9/netG_A2B.pth" --generator_B2A  "checkpoints/spa_res9/netG_B2A.pth"

python utils/evaluate_models.py --gt "datasets/scgan_complete/test/A" --models SPA_ResNet9="checkpoints/spa_res9/A" --kid_subset_size 100

python utils/evaluate_models.py --gt "datasets/scgan_complete/test/B" --models SPA_ResNet9="checkpoints/spa_res9/B" --kid_subset_size 100

# Deactivate the virtual environment
deactivate