#求所有细胞系相同的扰动类型
import numpy as np
import pandas as pd

file_name = 'trt_cp_2'
cell_types = ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC']

def load_data(file, cell):
    df = pd.read_csv(file + '/' + cell + '_trt_cp_LINCS_2.tsv', sep=',')
    ids = df['pert_id'].values
    return ids

com_list = load_data(file_name, cell_types[0])
for cell in cell_types:
    pert_ids = load_data(file_name, cell)
    com_list = [id_ for id_ in pert_ids if id_ in com_list]
np.save(file_name + '/common_perts/' + 'all_common.npy', np.asarray(com_list))
