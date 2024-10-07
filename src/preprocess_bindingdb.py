import pandas as pd
# from DeepPurpose import dataset

PATH = '/home/debnathk/gramseq/'

df_bindingdb = pd.read_csv(PATH + 'data/BindingDB/BindingDB_All_202407.tsv', sep='\t', on_bad_lines='skip')
df_bindingdb.head().to_csv(PATH + 'data/BindingDB/preprocessed/bindingdb_head.csv', index=False)
print('Done!')

# X_names, X_smiles, X_targets, y = dataset.process_BindingDB(path= PATH + 'data/BindingDB/BindingDB_All_202407.tsv', y='Kd', binary = False, \
# 					convert_to_log = True, threshold = 30)

# df_bindingdb = pd.DataFrame({'name': X_names, 'smiles': X_smiles, 'target sequence': X_targets, 'affinity': y})
# df_bindingdb.to_csv(PATH + 'data/BindingDB/preprocessed/bindingdb.csv', index=False)