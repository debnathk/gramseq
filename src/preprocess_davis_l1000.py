import pandas as pd

# Load davis dataset - train
df_davis_train = pd.read_csv('../data/davis (from FusionDTA)/davis_train.csv')

# Load davis dataset - test
df_davis_test = pd.read_csv('../data/davis (from FusionDTA)/davis_test.csv')

# Load compound_info dataset for L1000 project
df_l1000_cinfo = pd.read_csv('../data/compoundinfo_beta.csv')

# Filter dataset - train set
df_davis_train_filtered = df_davis_train[df_davis_train['compound_iso_smiles'].isin(df_l1000_cinfo['canonical_smiles'])]
print(df_davis_train_filtered)
# print(f'Unique common drugs - train set - davis: {len(set(df_davis_train_filtered))}')

# Filter dataset - test set
df_davis_test_filtered = df_davis_test[df_davis_test['compound_iso_smiles'].isin(df_l1000_cinfo['canonical_smiles'])]
print(df_davis_test_filtered)
# print(f'Unique common drugs - test set - davis: {len(set(df_davis_test_filtered))}')

# Take the outer join of the train and test sets
total_unique = pd.merge(df_davis_train_filtered, df_davis_test_filtered, how='outer')
print(f'Unique common drugs - davis: {len(set(total_unique["compound_iso_smiles"]))}')
