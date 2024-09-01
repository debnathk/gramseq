## Installations

For GPU support (in Windows Native)
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# Anything above 2.10 is not supported on the GPU on Windows Native
pip install "tensorflow<2.11"
# Verify the installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Others pip installations
```
pip install rdkit nltk pandas numpy==1.26.4 scikit-learn matplotlib prettytable ipykernel
```

## Dataset summary: 
### BindingDB dataset (Preprocessed)
```
No of unique drugs: 22381
No of unique targets: 1860
No of total interactions: 91751
```


