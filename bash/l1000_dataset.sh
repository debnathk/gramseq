#!/bin/bash
#SBATCH --job-name=l1000_dataset
#SBATCH --output output_l1000_datasets.log
#SBATCH --error error_l1000_datasets.log
#SBATCH --partition gpu
#SBATCH --gres=gpu:40g:1
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
source /home/debnathk/gramseq/.venv/bin/activate

# pip install DeepPurpose --quiet
# pip install git+https://github.com/bp-kelley/descriptastorus --quiet


# Navigate to the directory containing your Python script
# cd /path/to/your/python/script

# Run your Python script
# python src/l1000_datasets.py
python src/dataset_summary.py

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"