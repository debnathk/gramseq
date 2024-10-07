#!/bin/bash

# Ensure the script takes in a mode argument
if [ -z "$1" ]; then
    echo "Usage: $0 {train|test}"
    exit 1
fi

PROT_ENCODING='cnn'

MODE=$1
OUTPUT_LOG="../result_logs/output_gvae_dense_${PROT_ENCODING}_${MODE}.log"
ERROR_LOG="../result_logs/error_gvae_dense_${PROT_ENCODING}_${MODE}.log"

# Run the Python script with the specified mode, redirecting stdout and stderr
python ../src/run_gvae_dense_cnn.py --mode $MODE > $OUTPUT_LOG 2> $ERROR_LOG
