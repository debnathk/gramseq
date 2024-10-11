import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
import pandas as pd
import numpy as np
import utils
from predictor import DLEPS
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import argparse
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import time
from keras.callbacks import ModelCheckpoint
warnings.filterwarnings("ignore")

PATH  = '/home/debnathk/gramseq/'

# Add commandline arguments
parser = argparse.ArgumentParser(description='Add commandline arguments')
parser.add_argument('--dataset', type=str, choices=['bindingdb', 'davis', 'kiba'], required=True)
parser.add_argument('--rnaseq', action='store_true', help="Enable RNA-seq processing.")
parser.add_argument('--processed', action='store_true', help='Select processed dataset')
parser.add_argument('--protenc', type=str, choices=['CNN', 'RNN'], required=True)
parser.add_argument('--epochs', type=int, default=500, required=False)
parser.add_argument('--bs', type=int, default=256, required=False)
parser.add_argument('--folds', type=int, default=5, required=False)

# Parse arguments
args = parser.parse_args()
if args.dataset == 'bindingdb':
    if args.processed:
        df = pd.read_csv(PATH + 'data/l1000/l1000_bindingdb.csv')
        X_smiles, X_targets, y = df['smiles'], df['target sequence'], df['affinity']
    else:
        # Preprocess bindingdb dataset
        # X_names, X_smiles, X_targets, y = dataset.process_BindingDB(path= PATH + 'data/bindingdb/BindingDB_All_202407.tsv', y='Kd', binary = False, \
        #                     convert_to_log = True, threshold = 30)
        # df = pd.DataFrame({'name': X_names, 'smiles': X_smiles, 'target sequence': X_targets, 'affinity': y})
        # df.to_csv(PATH + 'data/bindingdb/preprocessed/bindingdb.csv', index=False)
        # Read Preprocessed davis dataset
        df = pd.read_csv(PATH + 'data/bindingdb/preprocessed/bindingdb.csv')
        X_smiles, X_targets, y = df['smiles'], df['target sequence'], df['affinity']
    print('Dataset summary: BindingDB dataset')
    print(f'No of unique drugs: {len(set(X_smiles))}')
    print(f'No of unique targets: {len(set(X_targets))}')
    print(f'No of total interactions: {len(X_smiles)}')

elif args.dataset == 'davis':
    if args.processed:
        df = pd.read_csv(PATH + 'data/l1000/l1000_davis.csv')
        X_smiles, X_targets, y = df['smiles'], df['target sequence'], df['affinity']
    else:
        # Preprocess davis dataset
        df = pd.read_csv(PATH + 'data/davis/preprocessed/davis.csv')
        X_smiles, X_targets, y = df['smiles'], df['target sequence'], df['affinity']
    print('Dataset summary: Davis dataset')
    print(f'No of unique drugs: {len(set(X_smiles))}')
    print(f'No of unique targets: {len(set(X_targets))}')
    print(f'No of total interactions: {len(X_smiles)}')

elif args.dataset == 'kiba':
    if args.processed:
        df = pd.read_csv(PATH + 'data/l1000/l1000_kiba.csv')
        X_smiles, X_targets, y = df['smiles'], df['target sequence'], df['affinity']
    else:
        df = pd.read_csv(PATH + 'data/kiba/preprocessed/kiba.csv')
        X_smiles, X_targets, y = df['smiles'], df['target sequence'], df['affinity']
    print('Dataset summary: KIBA dataset')
    print(f'No of unique drugs: {len(set(X_smiles))}')
    print(f'No of unique targets: {len(set(X_targets))}')
    print(f'No of total interactions: {len(X_smiles)}')

if args.rnaseq:
    # Calculate L1000
    df_l1000, rna_seq_vectors = utils.process_l1000(path=PATH, dataset=args.dataset)
    X_smiles, X_targets, y = df_l1000['smiles'], df_l1000['target sequence'], df_l1000['affinity']

    # Convert drugs to series object
    X_smiles = pd.Series(X_smiles)
    # One-hot encoding of drug SMILES
    S = pd.Series(X_smiles.unique()).apply(utils.smiles2onehot)
    S_dict = dict(zip(X_smiles.unique(), S))
    df_drugs = [S_dict[i] for i in X_smiles]
    one_hot_drugs = np.array(df_drugs)
    print(f'One-hot encoding of drug: {one_hot_drugs.shape}')

    # RNA-Seq vector encoding
    print(f'RNA-Seq vectors: {rna_seq_vectors.shape}')

    # Convert proteins to series object
    X_targets = pd.Series(X_targets)
    # One-hot encoding of proteins
    AA = pd.Series(X_targets.unique()).apply(utils.protein2onehot)
    AA_dict = dict(zip(X_targets.unique(), AA))
    df_proteins = [AA_dict[i] for i in X_targets]
    one_hot_proteins = np.array(df_proteins)
    print(f'One-hot encoding of protein: {one_hot_proteins.shape}')

    print(f'No of Labels: {y.shape}')

    # Clean data to remove inf, nan, if present any
    drugs = utils.clean_data(one_hot_drugs, fill_value=0)
    genes = utils.clean_data(rna_seq_vectors, fill_value=0)
    proteins = utils.clean_data(one_hot_proteins, fill_value=0)
    labels = utils.clean_data(y)

    # Split dataset
    drug_train, gene_train, protein_train, y_train, \
    drug_val, gene_val, protein_val, y_val, \
    drug_test, gene_test, protein_test, y_test = utils.train_val_test_split(X1=drugs, X2=genes, X3=proteins, y=labels)

    print(f'Train, Val, Test shapes - drug: {drug_train.shape, drug_val.shape, drug_test.shape}')
    print(f'Train, Val, Test shapes - gene: {gene_train.shape, gene_val.shape, gene_test.shape}')
    print(f'Train, Val, Test shapes - protein: {protein_train.shape, protein_val.shape, protein_test.shape}')
    print(f'Train, Val, Test shapes - y: {y_train.shape, y_val.shape, y_test.shape}')
else:
    # Convert drugs to series object
    X_smiles = pd.Series(X_smiles)
    # One-hot encoding of drug SMILES
    S = pd.Series(X_smiles.unique()).apply(utils.smiles2onehot)
    S_dict = dict(zip(X_smiles.unique(), S))
    df_drugs = [S_dict[i] for i in X_smiles]
    one_hot_drugs = np.array(df_drugs)
    print(f'One-hot encoding of drug: {one_hot_drugs.shape}')

    # Convert proteins to series object
    X_targets = pd.Series(X_targets)
    # One-hot encoding of proteins
    AA = pd.Series(X_targets.unique()).apply(utils.protein2onehot)
    AA_dict = dict(zip(X_targets.unique(), AA))
    df_proteins = [AA_dict[i] for i in X_targets]
    one_hot_proteins = np.array(df_proteins)
    print(f'One-hot encoding of protein: {one_hot_proteins.shape}')

    print(f'No of Labels: {y.shape}')

    # Clean data to remove inf, nan, if present any
    drugs = utils.clean_data(one_hot_drugs, fill_value=0)
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
dleps_p = DLEPS(rnaseq=args.rnaseq, protenc=args.protenc)
model = dleps_p.model[0]
print(model.summary())

# Train function
def main(): 
    # Start training
    tr_start = time.time()
    print("----START TRAINING----")

    # Hyperparameters
    # epochs = 100
    # batch_size = 512
    for fold in range(args.folds):
        # compile the model
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) 
        # Use ModelCheckpoint to save model and weights
        WEIGHTPATH = PATH + 'model_weights/'
        os.makedirs(WEIGHTPATH, exist_ok=True)
        if args.rnaseq:
            checkpoint = ModelCheckpoint(WEIGHTPATH + f'bs{args.bs}_ep{args.epochs}_rnaseq_{args.dataset}_{args.protenc}_fold{fold}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            # train the model
            early_stopping = EarlyStopping(monitor='val_mae', patience=100)
            history = model.fit([drug_train, gene_train, protein_train], y_train, batch_size=args.bs, epochs=args.epochs, callbacks=[checkpoint, early_stopping], validation_data=([drug_val, gene_val, protein_val], y_val))

            # Plot the training and validation loss
            PLOTSTPATH = PATH + 'results/plots/'
            os.makedirs(PLOTSTPATH, exist_ok=True)
            plt.title(f"Loss Curve: Drug Encoding=GVAE, RNA-Seq={args.rnaseq}, Protein Encoding={args.protenc}")
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(PLOTSTPATH + f"loss_bs{args.bs}_ep{args.epochs}_rnaseq_{args.dataset}_{args.protenc}_fold{fold}.png")
            plt.close()

            # Plot the training and validation MAE
            plt.title(f"MAE Curve: Drug Encoding=GVAE, RNA-Seq={args.rnaseq}, Protein Encoding={args.protenc}")
            plt.plot(history.history['mae'], label='Train MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.savefig(PLOTSTPATH + f"mae_bs{args.bs}_ep{args.epochs}_rnaseq_{args.dataset}_{args.protenc}_fold{fold}.png")
            plt.close()

        else:
            checkpoint = ModelCheckpoint(WEIGHTPATH + f'bs{args.bs}_ep{args.epochs}_{args.dataset}_{args.protenc}_fold{fold}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            # train the model
            early_stopping = EarlyStopping(monitor='val_mae', patience=100)
            history = model.fit([drug_train, protein_train], y_train, batch_size=args.bs, epochs=args.epochs, callbacks=[checkpoint, early_stopping], validation_data=([drug_val, protein_val], y_val))

            # Plot the training and validation loss
            PLOTSTPATH = PATH + 'results/plots/'
            os.makedirs(PLOTSTPATH, exist_ok=True)
            plt.title(f"Loss Curve: Drug Encoding=GVAE, RNA-Seq={args.rnaseq}, Protein Encoding={args.protenc}")
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(PLOTSTPATH + f"loss_bs{args.bs}_ep{args.epochs}_{args.dataset}_{args.protenc}_fold{fold}.png")
            plt.close()

            # Plot the training and validation MAE
            plt.title(f"MAE Curve: Drug Encoding=GVAE, RNA-Seq={args.rnaseq}, Protein Encoding={args.protenc}")
            plt.plot(history.history['mae'], label='Train MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.savefig(PLOTSTPATH + f"mae_bs{args.bs}_ep{args.epochs}_{args.dataset}_{args.protenc}_fold{fold}.png")
            plt.close()

            print("----END TRAINING----")
            tr_end = time.time()
            print(f'Elapsed time for training: {tr_end - tr_start}')

        # Start testing
        if args.rnaseq:
            print('----LOAD PRETRAINED MODEL----')
            WEIGHTPATH = PATH + 'model_weights/'
            model.load_weights(WEIGHTPATH + f'bs{args.bs}_ep{args.epochs}_rnaseq_{args.dataset}_{args.protenc}_fold{fold}.hdf5')
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
            # table = table.get_string()

            # Save the metrics
            TABLESPATH = PATH + 'results/tables/'
            os.makedirs(TABLESPATH, exist_ok=True)
            with open(TABLESPATH + f'bs{args.bs}_ep{args.epochs}_rnaseq_{args.dataset}_{args.protenc}_fold{fold}.txt', 'w') as f:
                f.write(f'{test_mse_loss},{test_pearson_corr},{test_c_index}')

            # Plot validation results
            plt.scatter(y_val, y_pred_val)
            m, b = np.polyfit(y_val, y_pred_val, 1)
            plt.plot(y_val, m * y_val + b, 'r')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted')
            plt.savefig(PLOTSTPATH + f"avsp_bs{args.bs}_ep{args.epochs}_rnaseq_{args.dataset}_{args.protenc}_fold{fold}.png")
            plt.close()

            # Plot validation results
            plt.scatter(y_test, y_pred_test)
            m, b = np.polyfit(y_test, y_pred_test, 1)
            plt.plot(y_test, m * y_test + b, 'r')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted')
            plt.savefig(PLOTSTPATH + f"avsp_bs{args.bs}_ep{args.epochs}_rnaseq_{args.dataset}_{args.protenc}_fold{fold}.png")
            plt.close()

        else:
            print('----LOAD PRETRAINED MODEL----')
            WEIGHTPATH = PATH + 'model_weights/'
            model.load_weights(WEIGHTPATH + f'bs{args.bs}_ep{args.epochs}_{args.dataset}_{args.protenc}_fold{fold}.hdf5')
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
            # table = table.get_string()

            # Save the metrics
            TABLESPATH = PATH + 'results/tables/'
            os.makedirs(TABLESPATH, exist_ok=True)
            with open(TABLESPATH + f'bs{args.bs}_ep{args.epochs}_{args.dataset}_{args.protenc}_fold{fold}.txt', 'w') as f:
                f.write(f'{test_mse_loss},{test_pearson_corr},{test_c_index}')

            # Plot validation results
            plt.scatter(y_val, y_pred_val)
            m, b = np.polyfit(y_val, y_pred_val, 1)
            plt.plot(y_val, m * y_val + b, 'r')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted')
            plt.savefig(PLOTSTPATH + f"avsp_bs{args.bs}_ep{args.epochs}_{args.dataset}_{args.protenc}_fold{fold}.png")
            plt.close()

            # Plot validation results
            plt.scatter(y_test, y_pred_test)
            m, b = np.polyfit(y_test, y_pred_test, 1)
            plt.plot(y_test, m * y_test + b, 'r')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted')
            plt.savefig(PLOTSTPATH + f"avsp_bs{args.bs}_ep{args.epochs}_{args.dataset}_{args.protenc}_fold{fold}.png")
            plt.close()
            
        print('----END TESTING----')

# def main(mode):
#     if mode == 'train':
#         train()
#     elif mode == 'test':
#         test()
#     else:
#         raise ValueError("Mode must be either 'train' or 'test'.")
    
if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser(description="train or test")
    # parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode to run: 'train' or 'test'")
    # parser.add_argument('--log', type=str, help="Log file name")
    # args = parser.parse_args()

    # # Save results to log file
    # logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run

    # Remove log files
    # PATH = '/home/debnathk/phd/projects/gramseq/results/bs512/bindingdb/rnaseq_false/gvae_rnn/'
    # output_path = 'output_bs512_bindingdb_rnn_fold5.log'
    # if os.path.exists(PATH + output_path):
    #     os.remove(PATH + output_path)
    # error_path = 'error_bs512_bindingdb_rnn_fold5.log'
    # if os.path.exists(PATH + error_path):
    #     os.remove(PATH + error_path)
