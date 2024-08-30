import numpy as np
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit import Chem, DataStructs
from functools import reduce
# from scipy.stats import pearsonr
import nltk
# from molecule_vae import xlength, get_zinc_tokenizer
import zinc_grammar
import warnings
import pandas as pd


def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles

def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    return i*1.0 / len(smiles)

'''
def to1hot(smiles):
    #may have errors because of false smiles
    _tokenize = get_zinc_tokenizer(zinc_grammar.GCFG)
    _parser = nltk.ChartParser(zinc_grammar.GCFG)
    _productions = zinc_grammar.GCFG.productions()
    _prod_map = {}
    for ix, prod in enumerate(_productions):
        _prod_map[prod] = ix
    MAX_LEN = 277
    _n_chars = len(_productions)
    smiles_rdkit = []
    iid = []
    for i in range(len(smiles)):
        try:
            smiles_rdkit.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles[ i ])))
            iid.append(i)
            #print(i)
        except:
            print("DLEPS: Error when process SMILES using rdkit at %d, skipped this molecule" % (i))
    assert type(smiles_rdkit) == list
    tokens = list(map(_tokenize, smiles_rdkit))
    parse_trees = []
    i = 0
    badi = []
    for t in tokens:
        #while True:
        try:
            tp = next(_parser.parse(t))
            parse_trees.append(tp)
        except:
            print("DLEPS: Parse tree error at %d, skipped this molecule" % i)
            badi.append(i)
        i += 1
        #print(i)
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([_prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, _n_chars), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        if num_productions > MAX_LEN:
            print("DLEPS: Large molecules, out of range, still proceed")
        
            one_hot[i][np.arange(MAX_LEN),indices[i][:MAX_LEN]] = 1.
        else:    
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.            
    return one_hot
'''

def get_fp(smiles):
    fp = []
    for mol in smiles:
        fp.append(mol2image(mol, n=2048))
    return fp


def mol2image(x, n=2048):
    try:
        m = Chem.MolFromSmiles(x)
        fp = Chem.RDKFingerprint(m, maxPath=4, fpSize=n)
        res = np.zeros(len(fp))
        DataStructs.ConvertToNumpyArray(fp, res)
        return res
    except:
        warnings.warn('Unable to calculate Fingerprint', UserWarning)
        return [np.nan]


def sanitize_smiles(smiles, canonize=True):
    """
    Takes list of SMILES strings and returns list of their sanitized versions.
    For definition of sanitized SMILES check http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
        Args:
            smiles (list): list of SMILES strings
            canonize (bool): parameter specifying whether to return canonical SMILES or not.

        Output:
            new_smiles (list): list of SMILES and NaNs if SMILES string is invalid or unsanitized.
            If 'canonize = True', return list of canonical SMILES.

        When 'canonize = True' the function is analogous to: canonize_smiles(smiles, sanitize=True).
    """
    new_smiles = []
    for sm in smiles:
        try:
            if canonize:
                new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=True)))
            else:
                new_smiles.append(sm)
        except: 
#warnings.warn('Unsanitized SMILES string: ' + sm, UserWarning)
            new_smiles.append('')
    return new_smiles


def canonize_smiles(smiles, sanitize=True):
    """
    Takes list of SMILES strings and returns list of their canonical SMILES.
        Args:
            smiles (list): list of SMILES strings
            sanitize (bool): parameter specifying whether to sanitize SMILES or not.
            For definition of sanitized SMILES check
            http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol

        Output:
            new_smiles (list): list of canonical SMILES and NaNs if SMILES string is invalid or unsanitized
            (when 'sanitize = True')

        When 'sanitize = True' the function is analogous to: sanitize_smiles(smiles, canonize=True).
    """
    new_smiles = []
    for sm in smiles:
        try:
            new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=sanitize)))
        except:
            warnings.warn(sm + ' can not be canonized: invalid SMILES string!', UserWarning)
            new_smiles.append('')
    return new_smiles


def save_smi_to_file(filename, smiles, unique=True):
    """
    Takes path to file and list of SMILES strings and writes SMILES to the specified file.

        Args:
            filename (str): path to the file
            smiles (list): list of SMILES strings
            unique (bool): parameter specifying whether to write only unique copies or not.

        Output:
            success (bool): defines whether operation was successfully completed or not.
       """
    if unique:
        smiles = list(set(smiles))
    else:
        smiles = list(smiles)
    f = open(filename, 'w')
    for mol in smiles:
        f.writelines([mol, '\n'])
    f.close()
    return f.closed


def read_smi_file(filename, unique=True):
    """
    Reads SMILES from file. File must contain one SMILES string per line
    with \n token in the end of the line.

    Args:
        filename (str): path to the file
        unique (bool): return only unique SMILES

    Returns:
        smiles (list): list of SMILES strings from specified file.
        success (bool): defines whether operation was successfully completed or not.

    If 'unique=True' this list contains only unique copies.
    """
    f = open(filename, 'r')
    molecules = []
    for line in f:
        molecules.append(line[:-1])
    if unique:
        molecules = list(set(molecules))
    else:
        molecules = list(molecules)
    f.close()
    return molecules, f.closed

def standardize_smiles(smiles):
    try:
        if smiles is not np.nan:
            return MolToSmiles(MolFromSmiles(smiles))
    except:
        return smiles
    
def smiles2onehot(s):

    def xlength(y):
        return reduce(lambda sum, element: sum + 1, y, 0)

    def get_zinc_tokenizer(cfg):
        long_tokens = [a for a in list(cfg._lexical_index.keys()) if xlength(a) > 1] ####
        replacements = ['$','%','^'] # ,'&']
        assert xlength(long_tokens) == len(replacements) ####xzw
        for token in replacements: 
            assert token not in cfg._lexical_index ####
        
        def tokenize(smiles):
            for i, token in enumerate(long_tokens):
                smiles = smiles.replace(token, replacements[i])
            tokens = []
            for token in smiles:
                try:
                    ix = replacements.index(token)
                    tokens.append(long_tokens[ix])
                except:
                    tokens.append(token)
            return tokens
        
        return tokenize
    
    _tokenize = get_zinc_tokenizer(zinc_grammar.GCFG)
    _parser = nltk.ChartParser(zinc_grammar.GCFG)
    _productions = zinc_grammar.GCFG.productions()
    _prod_map = {}
    for ix, prod in enumerate(_productions):
        _prod_map[prod] = ix
    MAX_LEN = 277
    _n_chars = len(_productions)
    one_hot = np.zeros((MAX_LEN, _n_chars), dtype=np.float32)

    token = map(_tokenize, s)
    tokens = []
    for t in token:
        tokens.append(t[0])
    tp = None
    try:
        tp = next(_parser.parse(tokens))
    except:
        # print("Parse tree error")
        return one_hot

    productions_seq = tp.productions()
    idx = np.array([_prod_map[prod] for prod in productions_seq], dtype=int)
    num_productions = len(idx)
    if num_productions > MAX_LEN:
            # print("Too large molecules, out of range")
            one_hot[np.arange(MAX_LEN),idx[:MAX_LEN]] = 1.
    else:    
            one_hot[np.arange(num_productions),idx] = 1.
            one_hot[np.arange(num_productions, MAX_LEN),-1] = 1.

    return one_hot

amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

from sklearn.preprocessing import OneHotEncoder
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))

MAX_SEQ_PROTEIN = 1000

def protein2onehot(x):
    temp = list(x.upper())
    temp = [i if i in amino_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_PROTEIN:
        temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
    else:
        temp = temp [:MAX_SEQ_PROTEIN]

    return enc_protein.transform(np.array(temp).reshape(-1,1)).toarray().T

def convert_y_unit(y, from_, to_):
	array_flag = False
	if isinstance(y, (int, float)):
		y = np.array([y])
		array_flag = True
	y = y.astype(float)    
	# basis as nM
	if from_ == 'nM':
		y = y
	elif from_ == 'p':
		y = 10**(-y) / 1e-9

	if to_ == 'p':
		zero_idxs = np.where(y == 0.)[0]
		y[zero_idxs] = 1e-10
		y = -np.log10(y*1e-9)
	elif to_ == 'nM':
		y = y
        
	if array_flag:
		return y[0]
	return y

import numpy as np

import numpy as np

def train_val_test_split(X1=None, X2=None, X3=None, y=None, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
    """
    Split the data into train, validation, and test sets.

    Parameters:
    X1 : array-like, shape (n_samples, n_features1)
        First feature matrix.
        
    X2 : array-like, shape (n_samples, n_features2)
        Second feature matrix.
    
    X3 : array-like, shape (n_samples, n_features3)
        Third feature matrix.

    y : array-like, shape (n_samples,)
        Target variable.

    train_size : float, optional (default=0.7)
        Proportion of the dataset to include in the train split.

    val_size : float, optional (default=0.15)
        Proportion of the dataset to include in the validation split.

    test_size : float, optional (default=0.15)
        Proportion of the dataset to include in the test split.

    random_state : int or RandomState instance or None, optional (default=None)
        Controls the shuffling applied to the data before splitting.

    Returns:
    X1_train, X2_train, X3_train, y_train : arrays
        Training set.

    X1_val, X2_val, X3_val, y_val : arrays
        Validation set.

    X1_test, X2_test, X3_test, y_test : arrays
        Test set.
    """
    # assert train_size + val_size + test_size == 1, "The sum of train_size, val_size, and test_size must equal 1."
    # assert np.isclose(train_size + val_size + test_size, 1.0), "The sum of train_size, val_size, and test_size must equal 1."

    # Compute sizes
    num_samples = len(X1)
    num_train = int(train_size * num_samples)
    num_val = int(val_size * num_samples)

    # Shuffle indices
    indices = np.arange(num_samples)
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(indices)

    # Split indices
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    # Split datasets
    if X2 is not None:
        X1_train, X2_train, X3_train = X1[train_indices], X2[train_indices], X3[train_indices]
        X1_val, X2_val, X3_val = X1[val_indices], X2[val_indices], X3[val_indices]
        X1_test, X2_test, X3_test = X1[test_indices], X2[test_indices], X3[test_indices]
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

        return X1_train, X2_train, X3_train, y_train, X1_val, X2_val, X3_val, y_val, X1_test, X2_test, X3_test, y_test
    
    else:
        X1_train, X3_train = X1[train_indices], X3[train_indices]
        X1_val, X3_val = X1[val_indices], X3[val_indices]
        X1_test, X3_test = X1[test_indices], X3[test_indices]
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

        return X1_train, X3_train, y_train, X1_val, X3_val, y_val, X1_test, X3_test, y_test


def clean_data(data, fill_value=0):
    """
    Fill NaN or infinite values in a 1D or 3D NumPy array.

    Parameters:
    data (np.ndarray): The input array to process. Can be 1D or 3D.
    fill_value (any): The value to use when filling NaN/infinite values in 3D data.

    Returns:
    np.ndarray: The array with NaN and infinite values filled.
    """
    if data.ndim == 1:
        # Replace infinite values with NaN for uniform processing
        data = np.where(np.isinf(data), np.nan, data)
        # Convert to pandas Series for easier handling
        series = pd.Series(data)
        # Fill NaN values with mean
        filled_series = series.fillna(series.mean())
        return filled_series.values
    elif data.ndim == 3:
        # Replace infinite values with NaN for uniform processing
        data = np.where(np.isinf(data), np.nan, data)
        
        n_samples, n_features, n_timesteps = data.shape
        filled_data = np.copy(data)
        
        for i in range(n_features):
            feature_data = data[:, i, :]
            df = pd.DataFrame(feature_data)
            # Fill NaN values with the constant value
            df_filled = df.fillna(fill_value)
            filled_data[:, i, :] = df_filled.values
        
        return filled_data
    else:
        raise ValueError("Data must be either 1D or 3D")
    
def extract_drug_name(input_string):
    # Split the string by underscores
    parts = input_string.split('_')
    
    # Check if there are at least five parts to avoid IndexError
    if len(parts) > 4:
        return parts[4]
    else:
        return "Error: Input string does not contain enough parts"
    
def l1000_data_preprocessing(cleaned_data):
     
     landmark_genes = pd.read_csv("../../data/landmark_genes.csv", header=None)
     
     return None


def pearson_correlation(y_pred, y_test):
    """
    Calculate the Pearson correlation coefficient between two arrays.

    Parameters:
    y_pred (array-like): Predicted values
    y_test (array-like): Actual values

    Returns:
    float: Pearson correlation coefficient
    float: p-value
    """
    # Calculate the Pearson correlation coefficient and the p-value
    correlation = np.corrcoef(y_pred, y_test)[0, 1]
    
    return correlation

def mse_loss(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (numpy array): Array of true values.
    y_pred (numpy array): Array of predicted values.

    Returns:
    float: The MSE value.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the squared differences
    squared_diffs = (y_true - y_pred) ** 2
    
    # Compute the mean of squared differences
    mse = np.mean(squared_diffs)
    
    return mse

def c_index(y_true, y_pred):
    """
    Calculate the Concordance Index (C-index) between true survival times and predicted risk scores.

    Parameters:
    y_true (numpy array): Array of true survival times.
    y_pred (numpy array): Array of predicted risk scores.

    Returns:
    float: The C-index value.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Initialize counters
    concordant_pairs = 0
    permissible_pairs = 0
    
    # Iterate through all pairs of subjects
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:  # Check if the pair is permissible
                permissible_pairs += 1
                # Check if the predictions are concordant
                if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                    concordant_pairs += 1
                elif y_pred[i] == y_pred[j]:
                    concordant_pairs += 0.5  # Tie in predictions are considered half-concordant

    # Calculate the C-index
    c_index = concordant_pairs / permissible_pairs if permissible_pairs > 0 else 0
    
    return c_index