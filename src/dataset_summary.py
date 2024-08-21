import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
import warnings
import pandas as pd
import numpy as np
import utils
from predictor_gvae_rnaseq_rnn import DLEPS
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import argparse
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import h5py
import logging
import time
warnings.filterwarnings("ignore")

print('\n-------------------------------Dataset Summary - BindingDB----------------------------\n')
# Preprocess bindingdb dataset
X_drugs, X_targets, y = dataset.process_BindingDB(path='/home/debnathk/phd/projects/gramseq/data/BindingDB/BindingDB_All_202407.tsv')
print(f'No of unique drugs: {len(set(X_drugs))}')
print(f'No of unique proteins: {len(set(X_targets))}')
print(f'Total interactions: {len(y)}')

print('\n-------------------------------Dataset Summary - Davis----------------------------\n')
# Preprocess davis dataset
X_drugs, X_targets, y = dataset.load_process_DAVIS()
print(f'No of unique drugs: {len(set(X_drugs))}')
print(f'No of unique proteins: {len(set(X_targets))}')
print(f'Total interactions: {len(y)}')

print('\n-------------------------------Dataset Summary - KIBA----------------------------\n')
# Preprocess KIBA dataset
X_drugs, X_targets, y = dataset.load_process_KIBA()
print(f'No of unique drugs: {len(set(X_drugs))}')
print(f'No of unique proteins: {len(set(X_targets))}')
print(f'Total interactions: {len(y)}')

