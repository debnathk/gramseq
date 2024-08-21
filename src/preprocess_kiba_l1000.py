import pandas as pd
from utils import standardize_smiles

# Load kiba dataset - train
df_kiba_train = pd.read_csv('../data/kiba (from FusionDTA)/kiba_train.csv')

# Load kiba dataset - test
df_kiba_test = pd.read_csv('../data/kiba (from FusionDTA)/kiba_test.csv')

# Load compound_info dataset for L1000 project
df_l1000_cinfo = pd.read_csv('../data/compoundinfo_beta.csv')

# Filter dataset - train set
std_smiles = df_kiba_train['compound_iso_smiles'].apply(standardize_smiles)
df_kiba_train_filtered = std_smiles[std_smiles.isin(df_l1000_cinfo['canonical_smiles'])]
# print(f'Unique common drugs - train set - kiba: {len(set(df_kiba_train_filtered))}')

# Filter dataset - test set
std_smiles = df_kiba_test['compound_iso_smiles'].apply(standardize_smiles)
df_kiba_test_filtered = std_smiles[std_smiles.isin(df_l1000_cinfo['canonical_smiles'])]
# print(f'Unique common drugs - test set - kiba: {len(set(df_kiba_test_filtered))}')

# Take the outer join of the train and test sets
total_unique = pd.merge(df_kiba_train_filtered, df_kiba_test_filtered, how='outer')
print(f'Unique common drugs - kiba: {len(set(total_unique["compound_iso_smiles"]))}')