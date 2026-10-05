#!/bin/bash
# Slurm job for Table 3, experiment 1 (CycleGAN, ResNet-9) on the USask Plato cluster.
#SBATCH --job-name=cycle_resnet          # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32GB                   # Memory required
#SBATCH --time=2-00:00:00              # Time limit, adjust as needed
#SBATCH --output=cycle_resnet_output.txt # Output file
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

# Train the CycleGAN model
python3 cycle_train.py --dataroot datasets/scgan_complete/ --cuda

# Test the model
python3 cycle_test.py --dataroot datasets/scgan_complete/ --cuda

# Note: cycle_test.py writes its outputs to checkpoints/cycle_res9_comp/{A,B}.
python utils/evaluate_models.py --gt "datasets/scgan_complete/test/A" --models Cycle_ResNet6="checkpoints/cycle_resnet6/A" --kid_subset_size 100

python utils/evaluate_models.py --gt "datasets/scgan_complete/test/B" --models Cycle_ResNet6="checkpoints/cycle_resnet6/B" --kid_subset_size 100

# Deactivate the virtual environment
deactivate