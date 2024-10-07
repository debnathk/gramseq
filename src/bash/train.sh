#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition gpu
#SBATCH --mem=128G     
#SBATCH --gres=gpu:1 

echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
# module purge
module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda  # Change this to the appropriate Anaconda version

# Activate your Python environment
source activate /lustre/home/debnathk/anaconda3/envs/gramseq

# pip install DeepPurpose --quiet
# pip install git+https://github.com/bp-kelley/descriptastorus --quiet

# Run your Python script
datasets=('bindingdb' 'davis' 'kiba')
protein_encodings=('CNN' 'RNN')
batch_sizes=(128 256 512)
folds=(1 5)
for dataset in "${datasets[@]}"
do 
    for encoding in "${protein_encodings[@]}"
    do 
        for bs in "${batch_sizes[@]}"
        do
            for fold in "${folds[@]}"
            do 
                python src/train.py --dataset $dataset\
                                    --protein-encoding $encoding\
                                    --batch-size $bs\
                                    --folds $fold > result_logs/output_train_test_bs${bs}_${encoding}_${dataset}_fold${fold}.log 2> result_logs/error_train_test_bs${bs}_${encoding}_${dataset}_fold${fold}.log
            done
        done
    done
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"