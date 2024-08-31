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

# Preprocess the davis dataset
# Load davis dataset - train
df_davis_train = pd.read_csv('../data/davis (from FusionDTA)/davis_train.csv')

# Load davis dataset - test
df_davis_test = pd.read_csv('../data/davis (from FusionDTA)/davis_test.csv')

# Load compound_info dataset for L1000 project
df_l1000_cinfo = pd.read_csv('../data/compoundinfo_beta.csv')

# Filter dataset - train set
df_davis_train_filtered = df_davis_train[df_davis_train['compound_iso_smiles'].isin(df_l1000_cinfo['canonical_smiles'])]
# print(f'Unique common drugs - train set - davis: {len(set(df_davis_train_filtered))}')

# Filter dataset - test set
df_davis_test_filtered = df_davis_test[df_davis_test['compound_iso_smiles'].isin(df_l1000_cinfo['canonical_smiles'])]
# print(f'Unique common drugs - test set - davis: {len(set(df_davis_test_filtered))}')

# Take the outer join of the train and test sets
total_unique = pd.merge(df_davis_train_filtered, df_davis_test_filtered, how='outer')
print(f'Unique common drugs - davis: {len(set(total_unique["compound_iso_smiles"]))}')

# train-test split
# Split drugs
X_drugs_train = df_davis_train_filtered['compound_iso_smiles']
X_drugs_test = df_davis_test_filtered['compound_iso_smiles']

# Split proteins
X_targets_train = df_davis_train_filtered['target_sequence']
X_targets_test = df_davis_test_filtered['target_sequence']

# Split y
y_train = df_davis_train_filtered['affinity']
y_test = df_davis_test_filtered['affinity']

# One-hot encoding of drug SMILES - train set
S = pd.Series(X_drugs_train.unique()).apply(utils.smiles2onehot)
S_dict = dict(zip(X_drugs_train.unique(), S))
df_drugs_train = [S_dict[i] for i in X_drugs_train]
one_hot_drugs_train = np.array(df_drugs_train)
print(f'One-hot encoding of drug - train set: {one_hot_drugs_train.shape}')

# One-hot encoding of drug SMILES - test set
S = pd.Series(X_drugs_test.unique()).apply(utils.smiles2onehot)
S_dict = dict(zip(X_drugs_test.unique(), S))
df_drugs_test = [S_dict[i] for i in X_drugs_test]
one_hot_drugs_test = np.array(df_drugs_test)
print(f'One-hot encoding of drug - test set: {one_hot_drugs_test.shape}')

# Generate l1000 data for drugs - train set
G = pd.Series(X_drugs_train.unique()).apply(utils.process_l1000)
G_dict = dict(zip(X_drugs_train.unique(), G))
df_genes_train = [G_dict[i] for i in X_drugs_train]
vector_genes_train = np.array(df_genes_train)
# vector_genes = utils.process_l1000(X_drugs[:10])
print(f'Vector encoding of gene expressions corresponding to drugs: {vector_genes_train.shape}')

# One-hot encoding of proteins - train set
AA = pd.Series(X_targets_train.unique()).apply(utils.protein2onehot)
AA_dict = dict(zip(X_targets_train.unique(), AA))
df_proteins_train = [AA_dict[i] for i in X_targets_train]
one_hot_proteins_train = np.array(df_proteins_train)
print(f'One-hot encoding of protein: {one_hot_proteins_train.shape}')

# One-hot encoding of proteins - test set
AA = pd.Series(X_targets_test.unique()).apply(utils.protein2onehot)
AA_dict = dict(zip(X_targets_test.unique(), AA))
df_proteins_test = [AA_dict[i] for i in X_targets_test]
one_hot_proteins_test = np.array(df_proteins_test)
print(f'One-hot encoding of protein: {one_hot_proteins_test.shape}')

print(f'No of Labels  - train set: {y_train.shape}')
print(f'No of Labels  - test set: {y_test.shape}')

'''
print("-----------------------Training - GVAE + RNA-Seq + RNN----------------------------")

# Clean data to remove inf, nan, if present any
drugs = utils.clean_data(one_hot_drugs, fill_value=0)
genes = utils.clean_data(vector_genes, fill_value=0)
proteins = utils.clean_data(one_hot_proteins, fill_value=0)
labels = utils.clean_data(y)

# Split dataset
drug_train, genes_train, protein_train, y_train, \
drug_val, genes_val, protein_val, y_val, \
drug_test, genes_test, protein_test, y_test = utils.train_val_test_split(X1=drugs, X2=genes, X3=proteins, y=labels)

print(f'Train, Val, Test shapes - drug: {drug_train.shape, drug_val.shape, drug_test.shape}')
print(f'Train, Val, Test shapes - genes: {genes_train.shape, genes_val.shape, genes_test.shape}')
print(f'Train, Val, Test shapes - protein: {protein_train.shape, protein_val.shape, protein_test.shape}')
print(f'Train, Val, Test shapes - y: {y_train.shape, y_val.shape, y_test.shape}')

# Load model
dleps_p = DLEPS()
model = dleps_p.model[0]
print(model.summary())

# Train function
def train(): 
    # Training start time
    tr_start = time.time()
    print("----START TRAINING----")
    # compile the model
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) 

    # Hyperparameters
    epochs = 500
    batch_size = 256

    # Use ModelCheckpoint to save model and weights
    from keras.callbacks import ModelCheckpoint
    filepath = "../model_weights/bs256_davis_rnaseq_rnn_fold1.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # train the model
    # early_stopping = EarlyStopping(monitor='val_mae', patience=10)
    history = model.fit([drug_train, genes_train, protein_train], y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint], validation_data=([drug_val, genes_val, protein_val], y_val))

    # Plot the training and validation loss
    import matplotlib.pyplot as plt
    plt.title("Loss Curve: Drug Encoding = GVAE, RNA-Seq Encoding = Dense, Protein Encoding = RNN")
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("../results/plots/loss_bs256_davis_rnaseq_rnn_fold1.png")
    plt.close()

    # Plot the training and validation MAE
    plt.title("MAE Curve: Drug Encoding = GVAE, RNA-Seq Encoding = Dense, Protein Encoding = RNN")
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig("../results/plots/mae_bs256_davis_rnaseq_rnn_fold1.png")
    plt.close()

    print("----END TRAINING----")
    tr_end = time.time()
    print(f'Elapsed time for training: {tr_end - tr_start}')

def test():
    print('----LOAD PRETRAINED MODEL----')
    model.load_weights("../model_weights/bs256_davis_rnaseq_rnn_fold1.hdf5")
    print('----PRETRAINED MODEL LOADED----')
    print('----START TESTING----')

    y_pred_val = model.predict([drug_val, genes_val, protein_val])
    y_pred_test = model.predict([drug_test, genes_test, protein_test])

    # Validation results
    val_mse_loss = utils.mse_loss(y_val, y_pred_val.ravel())
    val_pearson_corr = utils.pearson_correlation(y_val, y_pred_val.ravel())
    val_c_index = utils.c_index(y_val, y_pred_val.ravel())

    # Test results
    test_mse_loss = utils.mse_loss(y_test, y_pred_test.ravel())
    test_pearson_corr = utils.pearson_correlation(y_test, y_pred_test.ravel())
    test_c_index = utils.c_index(y_test, y_pred_test.ravel())

    table = PrettyTable()
    table.field_names = ["Metric", "Validation", "Test"]
    table.add_row(["MSE Loss", val_mse_loss, test_mse_loss])
    table.add_row(["Pearson Correlation", val_pearson_corr, test_pearson_corr])
    table.add_row(["Concordance Index", val_c_index, test_c_index])

    # Print results
    print(table)

    # Save the table
    table = table.get_string()
    with open('/home/debnathk/phd/projects/gramseq/results/bs256/davis/rnaseq_true/gvae_rnn/bs256_davis_rnaseq_rnn_fold1.txt', 'w') as f:
        f.write(table)

    # Plot validation results
    plt.scatter(y_val, y_pred_val)
    m, b = np.polyfit(y_val, y_pred_val, 1)
    plt.plot(y_val, m * y_val + b, 'r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

    # Plot validation results
    plt.scatter(y_test, y_pred_test)
    m, b = np.polyfit(y_test, y_pred_test, 1)
    plt.plot(y_test, m * y_test + b, 'r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()
    print('----END TESTING----')

def main(mode):
    if mode == 'train':
        train()
    elif mode == 'test':
        test()
    else:
        raise ValueError("Mode must be either 'train' or 'test'.")
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="train or test")
    # parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode to run: 'train' or 'test'")
    # parser.add_argument('--log', type=str, help="Log file name")
    # args = parser.parse_args()

    # # Save results to log file
    # logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run
    main('train')
    main('test')
'''