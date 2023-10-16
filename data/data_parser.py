import os

import cmapPy.pandasGEXpress.parse_gctx as pg
import pandas as pd

def data_process(pert_type, cell_id):
    # Phase 1
    sig_info1 = pd.read_csv("./GSE92742_Broad_LINCS_sig_info.txt", sep="\t")
    gene_info1 = pd.read_csv("./GSE92742_Broad_LINCS_gene_info.txt", sep="\t", dtype=str)
    landmark_gene_row_ids = gene_info1["pr_gene_id"][gene_info1["pr_is_lm"] == "1"]
    sig_info1 = sig_info1[(sig_info1["pert_type"] == pert_type)]
    sub_sig_info1 = sig_info1[(sig_info1["cell_id"] == cell_id)]
    sub_sig_info1.set_index("sig_id", inplace=True)

    gctoo1 = pg.parse("./GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", cid=sub_sig_info1.index.tolist(), rid=landmark_gene_row_ids)
    gctoo1.col_metadata_df = sub_sig_info1.copy()

    df_data_1 = gctoo1.data_df
    print(df_data_1)
    rids = df_data_1.index.tolist()
    print(rids)
    symbols = []
    for i in range(len(rids)):
        gene_symbol = gene_info1["pr_gene_symbol"][gene_info1["pr_gene_id"] == rids[i]].values[0]
        symbols.append(gene_symbol)

    with open("gene_symbols.csv", 'w+') as f:
        f.write('\n'.join(symbols))

    df_data_1 = df_data_1.transpose()
    #
    df_data_1["pert_id"] = gctoo1.col_metadata_df["pert_id"]
    df_data_1["pert_idose"] = gctoo1.col_metadata_df["pert_idose"]
    df_data_1["pert_itime"] = gctoo1.col_metadata_df["pert_itime"]

    df_data_1['pert_idose'] = df_data_1['pert_idose'].apply(lambda x:x[:-3])
    df_data_1['pert_itime'] = df_data_1['pert_itime'].apply(lambda x:x[:-2])

    df_data_1['itime_idose'] = df_data_1['pert_itime'].map(str) + '_' + df_data_1['pert_idose'].map(str)
    df_data_1['id'] = df_data_1['pert_id'].map(str) + '_' + \
                      df_data_1['pert_itime'].map(str) + '_' + \
                      df_data_1['pert_idose'].map(str)
    df_data_1 = df_data_1.drop(['pert_itime', 'pert_idose'], axis=1, inplace=False)
    df_data_1 = df_data_1.drop_duplicates().reset_index(drop=True)
    dir_1 = './'+pert_type+'_1'
    setDir(dir_1)
    df_data_1.to_csv(dir_1 + '/' + cell_id + '_' + pert_type + "_LINCS_1.tsv", sep='\t')

    # Phase 2
    sig_info2 = pd.read_csv("./GSE70138_Broad_LINCS_sig_info.txt", sep="\t")
    sig_info2 = sig_info2[(sig_info2["pert_type"] == pert_type)]
    sub_sig_info2 = sig_info2[(sig_info2["cell_id"] == cell_id)]
    sub_sig_info2.set_index("sig_id", inplace=True)
    gctoo2 = pg.parse("./GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx", cid=sub_sig_info2.index.tolist(), rid=landmark_gene_row_ids)
    gctoo2.col_metadata_df = sub_sig_info2.copy()
    #
    df_data_2 = gctoo2.data_df
    df_data_2.reindex(rids)
    print(df_data_2)
    df_data_2 = df_data_2.transpose()
    #
    df_data_2["pert_id"] = gctoo2.col_metadata_df["pert_id"]
    df_data_2["pert_idose"] = gctoo2.col_metadata_df["pert_idose"]
    df_data_2["pert_itime"] = gctoo2.col_metadata_df["pert_itime"]

    df_data_2['pert_idose'] = df_data_2['pert_idose'].apply(lambda x:x[:-3])
    df_data_2['pert_itime'] = df_data_2['pert_itime'].apply(lambda x:x[:-2])

    df_data_2['itime_idose'] = df_data_2['pert_itime'].map(str) + '_' + df_data_2['pert_idose'].map(str)
    df_data_2['id'] = df_data_2['pert_id'].map(str) + '_' + \
                      df_data_2['pert_itime'].map(str) + '_' + \
                      df_data_2['pert_idose'].map(str)
    df_data_2 = df_data_2.drop(['pert_itime', 'pert_idose'], axis=1, inplace=False)

    df_data_2 = df_data_2.drop_duplicates().reset_index(drop=True)
    df2_rid = rids
    df2_rid.append('pert_id')
    df2_rid.append('itime_idose')
    df2_rid.append('id')
    df_data_2 = df_data_2[df2_rid]
    dir_2 = './' + pert_type + '_2'
    setDir(dir_2)
    df_data_2.to_csv(dir_2 + '/' + cell_id+'_' + pert_type + "_LINCS_2.tsv", sep='\t')

def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)

pert_type = "trt_cp"
cell_list = ['A375', 'HA1E', 'HT29', 'MCF7', 'YAPC', 'HELA', 'PC3']
for cell in cell_list:
    data_process(pert_type, cell)