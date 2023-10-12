import pandas as pd
import numpy as np
np.set_printoptions(threshold=None)
df1 = pd.read_csv('./trt_cp_2/HT29_trt_cp_LINCS_1.tsv', sep='\t')
df2 = pd.read_csv('./trt_cp_2/MCF7_trt_cp_LINCS_1.tsv', sep='\t')

id1 = df1['id'].values
id2 = df2['id'].values
pert_id1 = df1['pert_id'].value_counts()
print(len(pert_id1))
pert_id2 = df2['pert_id'].value_counts()
print(len(pert_id2))
time_dose1 = df1['itime_idose'].value_counts()
print(time_dose1)
time_dose2 = df2['itime_idose'].value_counts()
print(time_dose2)
common_id = [id for id in id1 if id in id2]
print(len(common_id))
