import warnings
import pandas as pd
import numpy as np
import utils as ut
from DeepPurpose import utils, dataset
warnings.filterwarnings("ignore")

print('Processing BindingDB dataset...')
PATH = '/lustre/home/debnathk/gramseq/'
X_names, X_smiles, X_targets, y = dataset.process_BindingDB(path= PATH + 'data/BindingDB/BindingDB_All_202407.tsv', y='Kd', binary = False, \
					convert_to_log = True, threshold = 30)

df_bindingdb = pd.DataFrame({'name': X_names, 'smiles': X_smiles, 'target sequence': X_targets, 'affinity': y})
df_bindingdb.to_csv(PATH + 'data/BindingDB/preprocessed/bindingdb.csv', index=False)
print(df_bindingdb.head())
print(df_bindingdb.shape)