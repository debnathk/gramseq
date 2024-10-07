#!/bin/bash
#SBATCH --job-name=prepare_l1000
#SBATCH --output output_prepare_l1000.log
#SBATCH --error error_prepare_l1000.log
#SBATCH --partition cpu
#SBATCH --mem=128G      


echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
# module purge
# module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda  # Change this to the appropriate Anaconda version

# Activate your Python environment
source activate /lustre/home/debnathk/anaconda3/envs/gramseq

# pip install DeepPurpose --quiet
# pip install git+https://github.com/bp-kelley/descriptastorus --quiet


# Navigate to the directory containing your Python script
# cd /path/to/your/python/script

# Run your Python script
python src/prepare_l1000.py

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"