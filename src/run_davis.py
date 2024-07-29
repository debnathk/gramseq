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

# Preprocess Davis dataset
X_drugs, X_targets, y = dataset.load_process_DAVIS()
print('Drug 1: ' + X_drugs[0])
print('Target 1: ' + X_targets[0])
print('Score 1: ' + str(y[0]))

# Convert drugs to series object
X_drugs_series = pd.Series(X_drugs)

# One-hot encoding of drug SMILES
S = pd.Series(X_drugs_series.unique()).apply(utils.smiles2onehot)
S_dict = dict(zip(X_drugs_series.unique(), S))
df_drugs = [S_dict[i] for i in X_drugs]
one_hot_drugs = np.array(df_drugs)
print(f'One-hot encoding of drug: {one_hot_drugs.shape}')

# Generate l1000 data for drugs
vector_genes = utils.process_l1000(X_drugs)
print(f'Vector encoding of gene expressions corresponding to drugs: {vector_genes.shape}')

# Convert proteins to series object
X_targets = pd.Series(X_targets)

# One-hot encoding of proteins
AA = pd.Series(X_targets.unique()).apply(utils.protein2onehot)
AA_dict = dict(zip(X_targets.unique(), AA))
df_proteins = [AA_dict[i] for i in X_targets]
one_hot_proteins = np.array(df_proteins)
print(f'One-hot encoding of protein: {one_hot_proteins.shape}')

print(f'No of Labels: {y.shape}')

'''
print("-----------------------Training - GVAE + RNN----------------------------")

# Clean data to remove inf, nan, if present any
drugs = utils.clean_data(one_hot_drugs, fill_value=0)
# genes = utils.clean_data(genes, fill_value=0)
proteins = utils.clean_data(one_hot_proteins, fill_value=0)
labels = utils.clean_data(y)

# Split dataset
drug_train, protein_train, y_train, \
drug_val, protein_val, y_val, \
drug_test, protein_test, y_test = utils.train_val_test_split(X1=drugs, X3=proteins, y=labels)

print(f'Train, Val, Test shapes - drug: {drug_train.shape, drug_val.shape, drug_test.shape}')
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
    epochs = 100
    batch_size = 128

    # Use ModelCheckpoint to save model and weights
    from keras.callbacks import ModelCheckpoint
    filepath = "../model_weights/bs512_davis_rnn_fold5.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # train the model
    early_stopping = EarlyStopping(monitor='val_mae', patience=10)
    history = model.fit([drug_train, protein_train], y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, early_stopping], validation_data=([drug_val, protein_val], y_val))

    # Plot the training and validation loss
    import matplotlib.pyplot as plt
    plt.title("Loss Curve: Drug Encoding = GVAE, Protein Encoding = RNN")
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("../results/plots/loss_bs512_davis_rnn_fold5.png")
    plt.close()

    # Plot the training and validation MAE
    plt.title("MAE Curve: Drug Encoding = GVAE, Protein Encoding = RNN")
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig("../results/plots/mae_bs512_davis_rnn_fold5.png")
    plt.close()

    print("----END TRAINING----")
    tr_end = time.time()
    print(f'Elapsed time for training: {tr_end - tr_start}')

def test():
    print('----LOAD PRETRAINED MODEL----')
    model.load_weights("../model_weights/bs512_davis_rnn_fold5.hdf5")
    print('----PRETRAINED MODEL LOADED----')
    print('----START TESTING----')

    y_pred_val = model.predict([drug_val, protein_val])
    y_pred_test = model.predict([drug_test, protein_test])

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
    with open('/home/debnathk/phd/projects/gramseq/results/bs512/davis/rnaseq_false/gvae_rnn/bs512_davis_rnn_fold5.txt', 'w') as f:
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