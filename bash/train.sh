#!/bin/bash
#SBATCH --job-name=train
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

# Run your Python script
# datasets=('bindingdb' 'davis' 'kiba')
# protein_encodings=('CNN' 'RNN')
# batch_sizes=(128 256 512)
# epochs=(100 250 500)
# for dataset in "${datasets[@]}"
# do 
#     for encoding in "${protein_encodings[@]}"
#     do 
#         for bs in "${batch_sizes[@]}"
#         do
#             for ep in "${epochs[@]}"
#             do
#                 for fold in {1..5}
#                 do 
#                     python src/train.py --dataset $dataset\
#                                         --protein-encoding $encoding\
#                                         --batch-size $bs\
#                                         --epocs $ep\
#                                         --fold $fold > result_logs/output_train_test_bs${bs}_${encoding}_${dataset}_ep${ep}_fold${fold}.log 2> result_logs/error_train_test_bs${bs}_${encoding}_${dataset}_ep${ep}_fold${fold}.log
#                 done
#             done
#         done
#     done
# done
python src/train.py --dataset bindingdb\
                    --protenc CNN\
                    --rnaseq\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_bindingdb_rnaseq_cnn.log 2> error_bindingdb_rnaseq_cnn.log

python src/train.py --dataset bindingdb\
                    --protenc RNN\
                    --rnaseq\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_bindingdb_rnaseq_rnn.log 2> error_bindingdb_rnaseq_rnn.log

python src/train.py --dataset bindingdb\
                    --protenc CNN\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_bindingdb_cnn.log 2> error_bindingdb_cnn.log

python src/train.py --dataset bindingdb\
                    --protenc RNN\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_bindingdb_rnn.log 2> error_bindingdb_rnn.log

python src/process_tables.py

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"