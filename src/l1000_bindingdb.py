import re
import pandas as pd
import pickle
import json

PATH = '/home/debnathk/gramseq/'

def process_l1000(path):
    # Load l1000 data
    with open(PATH + 'data/l1000/l1000_vectors.pkl', 'rb') as file:
        data = pickle.load(file)
    file.close()

    # Load l1000 pert dictionary file
    with open(PATH + 'data/l1000/l1000_pert_dict.txt', 'r') as file:
        dict_l1000_pert = json.load(file)
    file.close()    

    # Load l1000 pert dictionary file
    with open(PATH + 'data/l1000/l1000_smiles_dict.txt', 'r') as file:
        dict_l1000_smiles = json.load(file)
    file.close()    

    # Load the bindingdb dataset
    # df_bindingdb = pd.read_csv(PATH + 'data/BindingDB/preprocessed/bindingdb.csv')
    df_bindingdb = pd.read_csv(path)

    # Create l1000 pert-smiles dataframe
    df_l1000_pert_smiles = pd.DataFrame({'pert': dict_l1000_smiles.keys(), 'smiles': dict_l1000_smiles.values()})

    # Create l1000 pert-idx dataframe
    df_l1000_pert_idx = pd.DataFrame({'pert': dict_l1000_pert.keys(), 'idx': dict_l1000_pert.values()})

    # Merge
    df_l1000_pert_smiles_idx = pd.merge(df_l1000_pert_smiles, df_l1000_pert_idx, on='pert')
    df_l1000_pert_smiles_idx.to_csv(PATH + 'data/l1000/l1000_pert_smiles_idx.csv', index=False)

    # Merge BindingDB and l1000
    df_l1000_bindingdb = pd.merge(df_bindingdb, df_l1000_pert_smiles_idx, on='smiles')
    df_l1000_bindingdb.drop_duplicates(subset=['smiles', 'target sequence'], inplace=True)
    df_l1000_bindingdb.to_csv(PATH + 'data/l1000/l1000_bindingdb.csv', index=False)

    # Extract l1000 data for filtered bindingdb dataset
    data_bindingdb = data[df_l1000_bindingdb['idx']]

    return data_bindingdb

data_bindingdb = process_l1000(PATH + 'data/BindingDB/preprocessed/bindingdb.csv')
# Select l1000 perts which are present in bindingb drug names
# common_smiles = [query for query in dict_l1000_smiles.values() for smiles in df_bindingdb['smiles'] if query == smiles]

# Filter df_bindingdb where only selected perts are present
# df_bindingdb = df_bindingdb[df_bindingdb['name'].str.contains(select_pert)]
# df_bindingdb = df_bindingdb[df_bindingdb['smiles'].str.contains(select_pert)]

# Join list elements with the '|' operator to create a regex pattern
# pattern = '|'.join(map(re.escape, [pert for pert in set(common_pert) if not pert.isdigit()]))
# pattern = '|'.join(map(re.escape, [pert for pert in set(common_smiles)]))

# Now use the pattern in the 'str.contains' method
# df_bindingdb = df_bindingdb[df_bindingdb['smiles'].str.contains(pattern, regex=True)]


# select_smiles = [query for query in set(common_smiles) for smiles in df_bindingdb['smiles'] if query == smiles]
# select_pert = {}
# for name in df_bindingdb['name']:
#     for pert in common_pert:
#         if pert in name:
#             select_pert[name] = pert
# print(select_pert)

# Select idxs of l1000 to integrate
# select_perts = [query for query in set(common_smiles) for pert in dict_l1000_smiles.keys() if query == pert]

# select_perts = []
# for pert, smiles in dict_l1000_smiles.items():
#     for query in select_smiles:
#         if smiles == query:
#             select_perts.append(pert)

# select_idx = [idx for s_pert in select_perts for pert, idx in dict_l1000_pert.items() if pert == s_pert]
# for s_pert in select_pert:
#     for pert, idx in dict_l1000.items():
#         if pert == s_pert:
#             select_idx.append(idx)


# for pert,idx in dict_l1000.items():
#     if not pert.isdigit():
#         for item in df_bindingdb['name']:
#             if pert in item:
#                 select_idx.append(idx)


# print(f'Shape of original BindingDB dataset: {df_bindingdb.shape}')
# print(f'No of unique drugs in original BindingDB dataset: {len(df_bindingdb["smiles"].unique())}')
# print(f'Shape of L1000 RNA-Seq data: {data.shape}')
# print(f'Shape of filtered BindingDB dataset: {df_l1000_bindingdb.shape}')
# print(f'No of unique drugs in filtered BindingDB dataset: {len(df_l1000_bindingdb["smiles"].unique())}')
print(f'Shape of L1000 RNA-Seq data for BindingDB dataset: {data_bindingdb.shape}')
# print(f'Length of the L1000 perturabgens dictionary: {len(dict_l1000)}')
# print(f'No of instances where a L1000 smiles is present in BindingDB drug names: {len(set(select_smiles))}')
# print(f'No of instances where a L1000 peturbagen is present in BindingDB drug names: {len(set(select_perts))}')
# print(f'No of indices where a L1000 peturbagen is present in BindingDB drug names:{len(set(select_idx))}')