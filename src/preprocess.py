import warnings
import pandas as pd
import numpy as np
import utils as ut
from DeepPurpose import utils, dataset
warnings.filterwarnings("ignore")

# Define path
PATH = '/lustre/home/debnathk/gramseq/'

# Preprocess L1000 dataset
# L1000 dataset can be downloaded from - https://lincs-dcic.s3.amazonaws.com/LINCS-sigs-2021/gmt/l1000_cp.gmt
print('Processing L1000 Chemical Perturbation RNA-Seq dataset...')
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

gmt_file_path = PATH + "data/l1000/l1000_cp.gmt"
gene_sets = read_gmt(gmt_file_path)

# Create dataframe
df_l1000 = pd.DataFrame({'pert_name': gene_sets.keys(), 'genes': gene_sets.values()})
genes_df = pd.DataFrame(df_l1000['genes'].to_list(), columns=[f'gene_{i+1}' for i in range(df_l1000['genes'].str.len().max())])
df_l1000 = pd.concat([df_l1000.drop(columns=['genes']), genes_df], axis=1)

# Filter instances with concentrations 10 uM
df_l1000 = df_l1000.loc[df_l1000['pert_name'].str.contains('10uM')]

# Save as csv
df_l1000.to_csv(PATH + 'data/l1000/l1000_cp.csv', index=False)
print(f'No of instances in L1000 Chemical Perturbation RNA-Seq dataset with 10 uM concentration: {len(df_l1000)}')

# Preprocess BindingDB
# BindingDB dataset can be downloaded from - https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_All_202409_tsv.zip
print('Processing BindingDB dataset...') 
X_names, X_smiles, X_targets, y = dataset.process_BindingDB(path= PATH + 'data/BindingDB/BindingDB_All_202407.tsv', y='Kd', binary = False, \
					convert_to_log = True, threshold = 30)

df_bindingdb = pd.DataFrame({'name': X_names, 'smiles': X_smiles, 'target sequence': X_targets, 'affinity': y})
df_bindingdb.to_csv(PATH + 'data/BindingDB/preprocessed/bindingdb.csv', index=False)

# print(df_bindingdb.head())
print('Dataset summary: BindingDB dataset (Preprocessed)')
print(f'No of unique drugs: {len(set(X_smiles))}')
print(f'No of unique targets: {len(set(X_targets))}')
print(f'No of total interactions: {len(X_smiles)}')

# Create RNA-Seq data: BindingDB dataset
print('Creating RNA-Seq data for BindingDB dataset...') 

# Extract up genes
df_up = df_l1000[df_l1000['pert_name'].str.contains('up')]

# Clean the drug names in the replicates - up
df_up['pert_name'] = df_up['pert_name'].apply(ut.extract_drug_name)

# Extract down genes
df_down = df_l1000[df_l1000['pert_name'].str.contains('down')]

# Clean the drug names in the replicates - down
df_down['pert_name'] = df_down['pert_name'].apply(ut.extract_drug_name)

print(f'No of unique perturbagens in L1000 dataset: {len(df_up["pert_name"].unique())}')

# Filter bindingdb drugs present in l1000 data
selected_names = []
for name in X_names:
    for pert in df_up['pert_name']: 
        if pert in name:
            selected_names.append(pert)

# Filter rows with selected names only
df_bindingdb = df_bindingdb.loc[df_bindingdb['name'].apply(lambda x: any(substring in x for substring in set(selected_names)))]
print(f'Filtered interactions in the BindingDB dataset: {len(df_bindingdb)}')
print(f'No of drugs in filtered interactions in the BindingDB dataset: {len(set(df_bindingdb["name"]))}')
print(f'No of targets in filtered interactions in the BindingDB dataset: {len(set(df_bindingdb["target sequence"]))}')

# Calculate
print('Load landmark genes...')
landmark_genes = pd.read_csv(PATH + 'data/landmark_genes.csv', header=None)

'''data_reg_list = []
for drug in selected_names:
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
print(f'Shape of vectors created: {data.shape}')'''

print('Finish!')