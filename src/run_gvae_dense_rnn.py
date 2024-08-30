import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import utils
from predictor_gvae_rnaseq_rnn import DLEPS
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import argparse
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import h5py
import logging
## Note: protobuf error can be resolved by downgrading it v3.20.1

# Load model
dleps_p = DLEPS()
model = dleps_p.model[0]
model.summary()

# Load dataset
h5f = h5py.File('../../data/drugs.h5', 'r')
drugs = h5f['data'][:]
h5f = h5py.File('../../data/genes.h5', 'r')
genes = h5f['data'][:]
h5f = h5py.File('../../data/proteins.h5', 'r')
proteins = h5f['data'][:]
h5f = h5py.File('../../data/pIC50.h5', 'r')
labels = h5f['data'][:]
print(drugs.shape)
print(genes.shape)
print(proteins.shape)
print(labels.shape)

# Clean data to remove inf, nan, if present any
drugs = utils.clean_data(drugs, fill_value=0)
genes = utils.clean_data(genes, fill_value=0)
proteins = utils.clean_data(proteins, fill_value=0)
labels = utils.clean_data(labels)

# Split dataset
drug_train, gene_train, protein_train, y_train, \
drug_val, gene_val, protein_val, y_val, \
drug_test, gene_test, protein_test, y_test = utils.train_val_test_split(drugs, genes, proteins, labels)

def train(): 
    print("----START TRAINING----")
    # compile the model
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) 

    # Use ModelCheckpoint to save model and weights
    from keras.callbacks import ModelCheckpoint
    # filepath = "test.weights.complete.rnn.best.hdf5"
    filepath = "test.weights.complete.rnn.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # train the model
    epochs = 100
    batch_size = 256
    early_stopping = EarlyStopping(monitor='val_mae', patience=10)
    history = model.fit([drug_train, gene_train, protein_train], y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, early_stopping], validation_data=([drug_val, gene_val, protein_val], y_val))

    # Plot the training and validation loss
    import matplotlib.pyplot as plt
    plt.title("Loss Curve: Drug Encoding = GVAE, Protein Encoding = RNN")
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./results/loss_gvae_rnn.png")
    plt.close()

    # Plot the training and validation MAE
    plt.title("MAE Curve: Drug Encoding = GVAE, Protein Encoding = RNN")
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig("./results/mae_gvae_rnn.png")
    plt.close()

    print("----END TRAINING----")

def test():
    print('----LOAD PRETRAINED MODEL----')
    model.load_weights("../../model_weights/test.weights.complete.rnn.best.hdf5")
    print('----PRETRAINED MODEL LOADED----')
    print('----START TESTING----')

    y_pred_val = model.predict([drug_val, gene_val, protein_val])
    y_pred_test = model.predict([drug_test, gene_test, protein_test])

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
    parser = argparse.ArgumentParser(description="train or test")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode to run: 'train' or 'test'")
    parser.add_argument('--log', type=str, help="Log file name")
    args = parser.parse_args()

    # Save results to log file
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run
    main(args.mode)
