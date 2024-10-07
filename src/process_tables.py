import os

PATH = '/home/debnathk/gramseq/results/tables/'

# List of txt files 
selected_files = [f for f in os.listdir(PATH) if 'CNN' in f and 'rnaseq' not in f and 'original' not in f]

with open(PATH + 'bs256_ep500_bindingdb_CNN.txt', 'w') as outfile:
    outfile.write('mse,pearson_corr,ci\n')
    for file in selected_files:
        with open(PATH + file, 'r') as f:
            line = f.readline().strip()
            outfile.write(line + '\n')

print("Merged values written to bs256_ep500_bindingdb_CNN.txt")

# List of txt files 
selected_files = [f for f in os.listdir(PATH) if 'RNN' in f and 'rnaseq' not in f and 'original' not in f]

with open(PATH + 'bs256_ep500_bindingdb_RNN.txt', 'w') as outfile:
    outfile.write('mse,pearson_corr,ci\n')
    for file in selected_files:
        with open(PATH + file, 'r') as f:
            line = f.readline().strip()
            outfile.write(line + '\n')

print("Merged values written to bs256_ep500_bindingdb_RNN.txt")

# List of txt files 
selected_files = [f for f in os.listdir(PATH) if 'CNN' in f and 'rnaseq' in f]

with open(PATH + 'bs256_ep500_rnaseq_bindingdb_CNN.txt', 'w') as outfile:
    outfile.write('mse,pearson_corr,ci\n')
    for file in selected_files:
        with open(PATH + file, 'r') as f:
            line = f.readline().strip()
            outfile.write(line + '\n')

print("Merged values written to bs256_ep500_rnaseq_bindingdb_CNN.txt")

# List of txt files 
selected_files = [f for f in os.listdir(PATH) if 'RNN' in f and 'rnaseq' in f]

with open(PATH + 'bs256_ep500_rnaseq_bindingdb_RNN.txt', 'w') as outfile:
    outfile.write('mse,pearson_corr,ci\n')
    for file in selected_files:
        with open(PATH + file, 'r') as f:
            line = f.readline().strip()
            outfile.write(line + '\n')

print("Merged values written to bs256_ep500_rnaseq_bindingdb_RNN.txt")
