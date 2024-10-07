#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output output_bs512_davis_rnn_fold5.log
#SBATCH --error error_bs512_davis_rnn_fold5.log
#SBATCH --partition gpu
#SBATCH --mem=128G      
#SBATCH --gres=gpu:40g:1


echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
# module purge
module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda  # Change this to the appropriate Anaconda version

# Activate your Python environment
source /home/debnathk/phd/projects/gramseq/.venv/bin/activate

# pip install DeepPurpose --quiet
# pip install git+https://github.com/bp-kelley/descriptastorus --quiet


# Navigate to the directory containing your Python script
# cd /path/to/your/python/script

# Run your Python script
cd /home/debnathk/phd/projects/gramseq/src/
python run.py


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"