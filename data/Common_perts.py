#求所有细胞系相同的扰动类型
import numpy as np
import pandas as pd

# file_name = 'trt_cp'
file_name = 'trt_oe'
cell_types = ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC']

def load_data(file, cell):
    df = pd.read_csv(file + '/' + cell + '.csv', sep=',')
    ids = df['id'].values
    return ids

com_list = load_data(file_name, cell_types[0])
for cell in cell_types:
    pert_ids = load_data(file_name, cell)
    com_list = [id_ for id_ in pert_ids if id_ in com_list]
np.save(file_name + '/common_perts/' + 'all_common.npy', np.asarray(com_list))

# for p in range(len(cell_types)):
#     for q in range(p+1, len(cell_types)):
#         pert_ids_1 = load_data(file_name, cell_types[p])
#         pert_ids_2 = load_data(file_name, cell_types[q])
#         com_list = [id_ for id_ in pert_ids_1 if id_ in pert_ids_2]
#         np.save(file_name + '/common_perts/' + cell_types[p] + '_' + cell_types[q] + '_common.npy', np.asarray(com_list))
