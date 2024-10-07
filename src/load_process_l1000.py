import utils
# from DeepPurpose import dataset
import pandas as pd
import numpy as np
'''
import utils
import json
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def process_l100(smiles):
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
    # print(data.shape)

    # Load l1000 compound info 
    df_comp_info = pd.read_csv('/home/debnathk/phd/projects/gramseq/data/compoundinfo_beta.csv')
    # print(df_comp_info.head())
    # print(df_comp_info.shape)

    # Save cmap name and corresponding canonical smiles as dictionary
    dict_smiles = {}

    for smiles, cmap in zip(df_comp_info['canonical_smiles'], df_comp_info['cmap_name']):
        if cmap not in dict_smiles.values():
            dict_smiles[smiles] = cmap

    # with open('../data/l1000_smiles.txt', 'w') as f:
    #     json.dump(dict_smiles, f)


    # Test - extract the index of drugs from smiles
    # Read the smiles file
    # with open('../data/l1000_smiles.txt', 'r') as f:
    #     dict_smiles = json.load(f)
    # f.close()

    # Read the idx file
    # with open('../data/l1000_drugs.txt', 'r') as f:
    #     dict_drugs = json.load(f)
    # f.close()

    # smiles_list = ['CCNC(=O)CCC(N)C(O)=O', 'NC(CCCNC(N)=O)C(O)=O', 'CCCN(CCC)C1CCc2ccc(O)cc2C1', 'C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1']

    collected_data = []
    for smiles in smiles_list:
        try:
            cmap = dict_smiles[smiles]
            idx = dict_drugs[cmap]
            # print(cmap)
            # print(idx)
            # print(data[idx])
            collected_data.append(data[idx])
        except KeyError:
            print(f'SMILES {smiles} not present in L1000 dataset. Performing similarity analysis with present SMILES...')
            similarities = {}
            i = 0
            while i < len(smiles_list) and smiles_list[i] != smiles:
                mol = Chem.MolFromSmiles(smiles)
                mol_p = Chem.MolFromSmiles(smiles_list[i])
                ms = [mol, mol_p]
                fpgen = AllChem.GetRDKitFPGenerator()
                fps = [fpgen.GetFingerprint(x) for x in ms]
                similarity = DataStructs.TanimotoSimilarity(fps[0],fps[1])
                similarities[similarity]= smiles_list[i]
                i += 1
            # print(f'Similarities: {similarities}')
            # Choose the smiles with highest similarity
            best_sim = max(similarities.keys())
            sim_smiles = similarities[best_sim]
            # print(f'Best similarity: {best_sim} with SMILES: {sim_smiles}')
            sim_cmap = dict_smiles[sim_smiles]
            sim_idx = dict_drugs[sim_cmap]
            sim_data = data[idx] * best_sim
            # print(sim_cmap)
            # print(sim_idx)
            # print(sim_data)
            collected_data.append(sim_data)

    stacked_data = np.stack(collected_data)
    return stacked_data
'''
    
if __name__ == "__main__":

    landmark_genes = pd.read_csv('../data/landmark_genes.csv', header=None)
        # Load l1000 compound info 
    df_comp_info = pd.read_csv('../data/compoundinfo_beta.csv')

    fpath = '../data/'
    df_l1000 = pd.read_csv(fpath + 'l1000_cp_10uM_all.csv')
    # print(df_l1000.head())
    # print(df_l1000.shape)

    # Count unique drugs with 10 uM concentration
    comp_list_10uM = []

    for i in range(len(df_l1000)):
        comp = df_l1000['0'][i].split()[0].split('_')[-2]
        comp_list_10uM.append(comp)

    # print(f'\nNo of unique perturbagens in L1000 dataset with 10 uM concentration: {len(set(comp_list_10uM))}\n')

    # Extract up genes
    df_up = df_l1000[df_l1000['0'].str.contains('up')]
    # print(df_l1000_up.head())

    # Clean the drug names in the replicates - up
    df_up['0'] = df_up['0'].apply(utils.extract_drug_name)
    # print(df_l1000_up.head())

    # Extract down genes
    df_down = df_l1000[df_l1000['0'].str.contains('down')]
    # print(df_l1000_down.head())

    # Clean the drug names in the replicates - down
    df_down['0'] = df_down['0'].apply(utils.extract_drug_name)

        # Save cmap name and corresponding canonical smiles as dictionary
    # dict_smiles = {}
    drug_list = []

    for drug, cmap in zip(df_down['0'], df_comp_info['cmap_name']):
        if drug==cmap:
            drug_list.append(drug)

    smiles = ['CCNC(=O)CCC(N)C(O)=O', 'NC(CCCNC(N)=O)C(O)=O', 'CCCN(CCC)C1CCc2ccc(O)cc2C1']
    data_reg_list = []
    for drug in drug_list[:5]:
        drug_count = 0
        df_reg = landmark_genes
        df_reg['up'] = [0] * 978
        df_reg['down'] = [0] * 978
        for drug_name in df_down['0']:
            if drug_name == drug:
                drug_count += 1
        filtered_up = df_up[df_up['0'] == drug]
        filtered_down = df_down[df_down['0'] == drug]
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
    print(data.shape)


