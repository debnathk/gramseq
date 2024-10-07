#!/bin/bash
#SBATCH --job-name=train_original
#SBATCH --partition gpu
#SBATCH --mem=128G     
#SBATCH --gres=gpu:1 
#SBATCH --time=1000:00:00

echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
# module purge
module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda  # Change this to the appropriate Anaconda version

# Activate your Python environment
source /home/debnathk/gramseq/.venv/bin/activate

# pip install DeepPurpose --quiet
# pip install git+https://github.com/bp-kelley/descriptastorus --quiet

mkdir -p result_logs

python src/train_original.py --dataset bindingdb\
                    --protenc CNN\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_bindingdb_original_cnn.log 2> error_bindingdb_original_cnn.log

python src/train_original.py --dataset bindingdb\
                    --protenc RNN\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_bindingdb_original_rnn.log 2> error_bindingdb_original_rnn.log

python src/train_original.py --dataset davis\
                    --protenc CNN\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_davis_original_cnn.log 2> error_davis_original_cnn.log

python src/train_original.py --dataset davis\
                    --protenc RNN\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_davis_original_rnn.log 2> error_davis_original_rnn.log

python src/train_original.py --dataset kiba\
                    --protenc RNN\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_kiba_original_rnn.log 2> error_kiba_original_rnn.log

python src/train_original.py --dataset kiba\
                    --protenc RNN\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_kiba_original_rnn.log 2> error_kiba_original_rnn.log

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"