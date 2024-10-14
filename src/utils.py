import numpy as np
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit import Chem, DataStructs
from sklearn.preprocessing import OneHotEncoder
from functools import reduce
import nltk
import zinc_grammar
import warnings
import pandas as pd
import pickle
import json


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

    new_smiles = []
    for sm in smiles:
        try:
            new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=sanitize)))
        except:
            warnings.warn(sm + ' can not be canonized: invalid SMILES string!', UserWarning)
            new_smiles.append('')
    return new_smiles

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


def train_val_test_split(X1=None, X2=None, X3=None, y=None, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):

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

    if data.ndim == 1:
        # Replace infinite values with NaN
        data = np.where(np.isinf(data), np.nan, data)

        series = pd.Series(data)
        # Fill NaN values with mean
        filled_series = series.fillna(series.mean())
        return filled_series.values
    elif data.ndim == 3:
        # Replace infinite values with NaN
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

def pearson_correlation(y_pred, y_test):

    correlation = np.corrcoef(y_pred, y_test)[0, 1]
    
    return correlation

def mse_loss(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    squared_diffs = (y_true - y_pred) ** 2
    mse = np.mean(squared_diffs)
    
    return mse

def c_index(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    concordant_pairs = 0
    permissible_pairs = 0
    
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                permissible_pairs += 1
                if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                    concordant_pairs += 1
                elif y_pred[i] == y_pred[j]:
                    concordant_pairs += 0.5  # Tie in predictions are considered half-concordant

    c_index = concordant_pairs / permissible_pairs if permissible_pairs > 0 else 0
    
    return c_index

def read_gmt(file_path):
    gene_sets = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            gene_set_name = parts[0]
            # description = parts[1]
            genes = parts[2:]
            gene_sets[gene_set_name] = genes
            # {
            #     # "description": description,
            #     "genes": genes
            # }
    return gene_sets

def process_l1000(path=None, dataset=None):

    # PATH = '/home/debnathk/gramseq/'
    # Load l1000 data
    with open(path + 'data/l1000/l1000_vectors.pkl', 'rb') as file:
        data = pickle.load(file)
    file.close()

    # Load l1000 pert dictionary file
    with open(path + 'data/l1000/l1000_pert_dict.txt', 'r') as file:
        dict_l1000_pert = json.load(file)
    file.close()    

    # Load l1000 pert dictionary file
    with open(path + 'data/l1000/l1000_smiles_dict.txt', 'r') as file:
        dict_l1000_smiles = json.load(file)
    file.close()    

    # Load the bindingdb dataset
    # df_bindingdb = pd.read_csv(PATH + 'data/BindingDB/preprocessed/bindingdb.csv')
    df = pd.read_csv(path + f'data/{dataset}/preprocessed/{dataset}.csv')
    if dataset != 'bindingdb':
        df['smiles'] = df['smiles'].apply(iso2can_smiles)

    # Create l1000 pert-smiles dataframe
    df_l1000_pert_smiles = pd.DataFrame({'pert': dict_l1000_smiles.keys(), 'smiles': dict_l1000_smiles.values()})

    # Create l1000 pert-idx dataframe
    df_l1000_pert_idx = pd.DataFrame({'pert': dict_l1000_pert.keys(), 'idx': dict_l1000_pert.values()})

    # Merge
    df_l1000_pert_smiles_idx = pd.merge(df_l1000_pert_smiles, df_l1000_pert_idx, on='pert')
    df_l1000_pert_smiles_idx.to_csv(path + 'data/l1000/l1000_pert_smiles_idx.csv', index=False)

    # Merge BindingDB and l1000
    df_l1000_merged = pd.merge(df, df_l1000_pert_smiles_idx, on='smiles')
    df_l1000_merged.drop_duplicates(subset=['smiles', 'target sequence'], inplace=True)
    df_l1000_merged.to_csv(path + f'data/l1000/l1000_{dataset}.csv', index=False)

    # Extract l1000 data for filtered bindingdb dataset
    data_dataset = data[df_l1000_merged['idx']]

    return df_l1000_merged, data_dataset

def iso2can_smiles(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smile}")
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        return can_smiles
    except Exception as e:
        print(f"Error converting SMILES ")