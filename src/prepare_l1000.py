import warnings
import pandas as pd
import numpy as np
import utils
import pickle
import json
warnings.filterwarnings("ignore")

# Define path
# PATH = '/lustre/home/debnathk/gramseq/'
PATH = '/home/debnathk/gramseq/'

print('Preparing L1000 Chemical Perturbation RNA-Seq dataset...')
gmt_file_path = PATH + "data/l1000/l1000_cp.gmt"
gene_sets = utils.read_gmt(gmt_file_path)

# Create dataframe
df_l1000 = pd.DataFrame({'pert_name': gene_sets.keys(), 'genes': gene_sets.values()})
genes_df = pd.DataFrame(df_l1000['genes'].to_list(), columns=[f'gene_{i+1}' for i in range(df_l1000['genes'].str.len().max())])
df_l1000 = pd.concat([df_l1000.drop(columns=['genes']), genes_df], axis=1)

# Filter instances with concentrations 10 uM
print('Filtering instances with perturbagen concentration = 10 uM')
df_l1000 = df_l1000.loc[df_l1000['pert_name'].str.contains('10uM')]

# Mapping perturbagen indices
print('Mapping perturgen indices in the L1000 dataset...')
dict_pert_name = {}
for idx, pert_name in enumerate(df_l1000['pert_name'].apply(utils.extract_drug_name).unique()):
    dict_pert_name[pert_name] = idx

# Save as csv
print('Saving mapped perturbagen names as data/l1000/l1000_pert_dict.txt')
# with open(PATH + 'data/l1000/l1000_pert_dict.txt', 'w') as file:
#     file.write(str(dict_pert_name))
with open(PATH + 'data/l1000/l1000_pert_dict.txt', 'w') as file:
    json.dump(dict_pert_name, file, indent=4)
file.close()

# Load the L1000 compoundinfo 
df_cinfo = pd.read_csv(PATH + 'data/compoundinfo_beta.csv')
dict_smiles = {}
for name, smiles in zip(df_cinfo['cmap_name'], df_cinfo['canonical_smiles']):
    dict_smiles[name] = smiles

# Save as csv
print('Saving mapped perturbagen smiles as data/l1000/l1000_smiles_dict.txt')
# with open(PATH + 'data/l1000/l1000_pert_dict.txt', 'w') as file:
#     file.write(str(dict_pert_name))
with open(PATH + 'data/l1000/l1000_smiles_dict.txt', 'w') as file:
    json.dump(dict_smiles, file, indent=4)
file.close()
'''
# Save as csv
print('Saving L1000 dataset data/l1000/l1000_cp.csv')
df_l1000.to_csv(PATH + 'data/l1000/l1000_cp.csv', index=False)
print(f'No of instances in L1000 Chemical Perturbation RNA-Seq dataset with 10 uM concentration: {len(df_l1000)}')

# Create RNA-Seq data
print('Vectorizing RNA-Seq profile for each perturbagen..') 

# Extract up genes
df_up = df_l1000[df_l1000['pert_name'].str.contains('up')]
# Clean the drug names in the replicates - up
df_up['pert_name'] = df_up['pert_name'].apply(utils.extract_drug_name)

# Extract down genes
df_down = df_l1000[df_l1000['pert_name'].str.contains('down')]
# Clean the drug names in the replicates - down
df_down['pert_name'] = df_down['pert_name'].apply(utils.extract_drug_name)

print(f'No of unique perturbagens in L1000 dataset: {len(df_up["pert_name"].unique())}')

# Calculate
print('Load landmark genes...')
landmark_genes = pd.read_csv(PATH + 'data/landmark_genes.csv', header=None)

data_reg_list = []
for drug in list(dict_pert_name.keys()):
    drug_count = 0
    df_reg = landmark_genes
    df_reg['up'] = [0] * 978
    df_reg['down'] = [0] * 978
    for drug_name in df_down['pert_name']:
        if drug_name == drug:
            drug_count += 1
    filtered_up = df_up[df_up['pert_name'] == drug]
    filtered_down = df_down[df_down['pert_name'] == drug]
    array_up = filtered_up.iloc[:, 1:].values
    array_up = array_up.flatten()
    array_down = filtered_down.iloc[:, 1:].values
    array_down = array_down.flatten()
    for item in array_up:
        df_reg.loc[df_reg[0] == item, 'up'] += 1
    for item in array_down:
        df_reg.loc[df_reg[0] == item, 'down'] += 1
    df_reg = df_reg.iloc[:, 1:] / drug_count
    df_reg = df_reg.values
    data_reg_list.append(df_reg)

data = np.stack(data_reg_list)
print(f'Shape of vectors created: {data.shape}')
print(data[:5])

# Save as pkl
print('Saving the vectors as data/l1000/l1000_vectors.pkl')
with open(PATH + 'data/l1000/l1000_vectors.pkl', 'wb') as file:
    pickle.dump(data, file)
file.close()
'''

print('Completed!')