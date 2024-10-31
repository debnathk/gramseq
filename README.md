## Installations

For GPU support (in Windows Native):
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# Anything above 2.10 is not supported on the GPU on Windows Native
pip install "tensorflow<2.11"
# Verify the installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Others pip installations (using the `requirements.txt` file):
```
pip install -r requirements.txt
```
The environment can also be setup using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate gramseq
```

## Data Preparation
Preparing the RNA-Seq data from L1000 project:
```
python src/prepare_l1000.py
```

## Training
To integrate RNA-seq data with the BindingDB dataset for model training, the following code can be used:
```
python src/train.py --dataset bindingdb\
                    --protenc CNN\
                    --rnaseq\
                    --epochs 500\
                    --folds 5 > ./result_logs/output_bindingdb_rnaseq_cnn.log 2> error_bindingdb_rnaseq_cnn.log
```
Similarly, the model can be trained on the Davis and KIBA datasets by replacing the `bindingdb` with `davis` or `kiba` for the `--dataset` argument, respectively. To use RNN as the protein encoder, `CNN` can be replaced with `RNN` and passed into the `--protenc` argument.
