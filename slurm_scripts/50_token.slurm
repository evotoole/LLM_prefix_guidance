#!/bin/bash
#SBATCH --job-name=llama3-70b-inf
#SBATCH --output=llama3-70b.out
#SBATCH --error=llama3-70b.err
#SBATCH --time=30:00:00                # Adjust as needed
#SBATCH --gres=gpu:a100:3              # Number/type of GPUs (adjust as needed)
#SBATCH --cpus-per-task=8             # Adjust for your workload
#SBATCH --mem=200G                     # Adjust for your workload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=evankarlotoole@gmail.com


# Load modules and activate your environment
module load python/3.10
source ~/env/bin/activate	 # Activate your virtual environment if you have one

# Optional: move to your working directory
cd /home/evotoole/scratch/llama3-70b-instruct

# Run your inference script
python3 /home/evotoole/large_llama/old_files/70B_script.py


