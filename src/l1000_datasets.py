from utils import process_l1000

PATH = '/home/debnathk/gramseq/'

l1000_bindingdb, _ = process_l1000(path=PATH, dataset='bindingdb')
l1000_davis, _ = process_l1000(path=PATH, dataset='davis')
l1000_kiba, _ = process_l1000(path=PATH, dataset='kiba')

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
print(f'Shape of L1000 RNA-Seq data for BindingDB dataset: {l1000_bindingdb.shape}')
print(f'Shape of L1000 RNA-Seq data for Davis dataset: {l1000_davis.shape}')
print(f'Shape of L1000 RNA-Seq data for KIBA dataset: {l1000_kiba.shape}')
# print(f'Length of the L1000 perturabgens dictionary: {len(dict_l1000)}')
# print(f'No of instances where a L1000 smiles is present in BindingDB drug names: {len(set(select_smiles))}')
# print(f'No of instances where a L1000 peturbagen is present in BindingDB drug names: {len(set(select_perts))}')
# print(f'No of indices where a L1000 peturbagen is present in BindingDB drug names:{len(set(select_idx))}')