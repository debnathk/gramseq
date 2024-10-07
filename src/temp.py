import utils
import pandas as pd
import numpy as np

# Load compound_info dataset for L1000 project
df_l1000_cinfo = pd.read_csv('../data/compoundinfo_beta.csv')

# Load davis dataset - train
df_davis_train = pd.read_csv('../data/davis (from FusionDTA)/davis_train.csv')
# Load davis dataset - test
df_davis_test = pd.read_csv('../data/davis (from FusionDTA)/davis_test.csv')

# Filter dataset - train
df_davis_train_cinfo = df_davis_train[df_davis_train['compound_iso_smiles'].isin(df_l1000_cinfo['canonical_smiles'])]
# Filter dataset - test
df_davis_test_cinfo = df_davis_test[df_davis_test['compound_iso_smiles'].isin(df_l1000_cinfo['canonical_smiles'])]

# Create compound_iso_smiles + cmap_name dictionary - train set
smi_cmap_dict_train = {}
for comp_smi in df_davis_train['compound_iso_smiles']:
    for idx, smi in enumerate(df_l1000_cinfo['canonical_smiles']):
        if comp_smi == smi:
            smi_cmap_dict_train[comp_smi] = df_l1000_cinfo['cmap_name'][idx]
print(smi_cmap_dict_train)

# Create the cmap list - train
cmap_list_train = []
for comp_smi, cmap in smi_cmap_dict_train.items():
    for smi in df_davis_train_cinfo['compound_iso_smiles']:
        if comp_smi == smi:
            cmap_list_train.append(cmap)

print(cmap_list_train)
print(len(cmap_list_train))

# Create compound_iso_smiles + cmap_name dictionary - test set
smi_cmap_dict_test = {}
for comp_smi in df_davis_test['compound_iso_smiles']:
    for idx, smi in enumerate(df_l1000_cinfo['canonical_smiles']):
        if comp_smi == smi:
            smi_cmap_dict_test[comp_smi] = df_l1000_cinfo['cmap_name'][idx]
print(smi_cmap_dict_test)

# Create the cmap list - test
cmap_list_test = []
for comp_smi, cmap in smi_cmap_dict_test.items():
    for smi in df_davis_test_cinfo['compound_iso_smiles']:
        if comp_smi == smi:
            cmap_list_test.append(cmap)

print(cmap_list_test)
print(len(cmap_list_test))

# Filter common cmap_name from l1000 dataset
df_l1000 = pd.read_csv('/home/debnathk/phd/projects/gramseq/data/l1000_cp_10uM_all.csv')

# Count unique drugs with 10 uM concentration
comp_list_10uM = []

for i in range(len(df_l1000)):
    comp = df_l1000['0'][i].split()[0].split('_')[-2]
    comp_list_10uM.append(comp)

# Extract up genes
df_l1000_up = df_l1000[df_l1000['0'].str.contains('up')]

# Clean the drug names in the replicates - up
df_l1000_up['0'] = df_l1000_up['0'].apply(utils.extract_drug_name)

# Extract down genes
df_l1000_down = df_l1000[df_l1000['0'].str.contains('down')]

# Clean the drug names in the replicates - down
df_l1000_down['0'] = df_l1000_down['0'].apply(utils.extract_drug_name)

# Load the landmark genes
landmark_genes = pd.read_csv('../data/landmark_genes.csv', header=None)

# Generate rna-seq data - train set
data_reg_list = []
for drug in cmap_list_train:
    drug_count = 0
    df_reg = landmark_genes
    df_reg['up'] = [0] * 978
    df_reg['down'] = [0] * 978
    for drug_name in df_l1000_down['0']:
        if drug_name == drug:
            drug_count += 1
    filtered_up = df_l1000_up[df_l1000_up['0'] == drug]
    filtered_down = df_l1000_down[df_l1000_down['0'] == drug]
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

data_train = np.stack(data_reg_list)
print(data_train.shape)

# Generate rna-seq data - test set
data_reg_list = []
for drug in cmap_list_test:
    drug_count = 0
    df_reg = landmark_genes
    df_reg['up'] = [0] * 978
    df_reg['down'] = [0] * 978
    for drug_name in df_l1000_down['0']:
        if drug_name == drug:
            drug_count += 1
    filtered_up = df_l1000_up[df_l1000_up['0'] == drug]
    filtered_down = df_l1000_down[df_l1000_down['0'] == drug]
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

data_test = np.stack(data_reg_list)
print(data_test.shape)

'''
def process_l1000(s):
    # Load l1000 dataset
    df_l1000 = pd.read_csv('/home/debnathk/phd/projects/gramseq/data/l1000_cp_10uM_all.csv')
    # print(df_l1000.head())
    # print(df_l1000.shape)

    # Count unique drugs with 10 uM concentration
    comp_list_10uM = []

    for i in range(len(df_l1000)):
        comp = df_l1000['0'][i].split()[0].split('_')[-2]
        comp_list_10uM.append(comp)

    # print(f'\nNo of unique perturbagens in L1000 dataset with 10 uM concentration: {len(set(comp_list_10uM))}\n')

    # Extract up genes
    df_l1000_up = df_l1000[df_l1000['0'].str.contains('up')]
    # print(df_l1000_up.head())

    # Clean the drug names in the replicates - up
    df_l1000_up['0'] = df_l1000_up['0'].apply(utils.extract_drug_name)
    # print(df_l1000_up.head())

    # Extract down genes
    df_l1000_down = df_l1000[df_l1000['0'].str.contains('down')]
    # print(df_l1000_down.head())

    # Clean the drug names in the replicates - down
    df_l1000_down['0'] = df_l1000_down['0'].apply(utils.extract_drug_name)
    # print(df_l1000_down.head())

    # Create dataset

    # Read the landmark_genes CSV file
    landmark_genes = pd.read_csv("../data/landmark_genes.csv", header=None)

    # Prepare a list to store the results
    data_reg_list = []

    # Precompute the count of occurrences for each drug in the down-regulated dataset
    drug_counts = df_l1000_up['0'].value_counts()

    # Iterate over each unique drug name and save the idx in dict
    dict_drugs = {}
    for drug, idx in zip(list(set(df_l1000_up['0'])), np.arange(len(set(df_l1000_up['0'])))):
        # Initialize the DataFrame for this drug with zeros for 'up' and 'down' counts
        if drug not in dict_drugs.keys():
            dict_drugs[drug] = int(idx)
        df_reg = landmark_genes.copy()
        df_reg['up'] = 0
        df_reg['down'] = 0

        # Filter the gene expression data for the current drug
        filtered_up = df_l1000_up[df_l1000_up['0'] == drug]
        filtered_down = df_l1000_down[df_l1000_down['0'] == drug]

        # Flatten the arrays of up-regulated and down-regulated genes
        array_up = filtered_up.iloc[:, 1:].to_numpy().flatten()
        array_down = filtered_down.iloc[:, 1:].to_numpy().flatten()

        # Count the occurrences of each gene in the 'up' and 'down' arrays
        up_counts = pd.Series(array_up).value_counts()
        down_counts = pd.Series(array_down).value_counts()

        # Update the 'up' and 'down' columns in the df_reg DataFrame
        df_reg['up'] = df_reg[0].map(up_counts).fillna(0)
        df_reg['down'] = df_reg[0].map(down_counts).fillna(0)

        # Normalize the counts by the total number of occurrences of the drug
        drug_count = drug_counts.get(drug, 1)  # Default to 1 to avoid division by zero
        df_reg.iloc[:, 1:] = df_reg.iloc[:, 1:] / drug_count

        # Convert the DataFrame to a numpy array and add to the results list
        data_reg_list.append(df_reg.iloc[:, 1:].to_numpy())

    # Save the unique drugs as dictionary
    # with open('../data/l1000_drugs.txt', 'w') as f:
    #     json.dump(dict_drugs, f)

    data = np.stack(data_reg_list)

    return data

print(process_l1000(smiles))
'''