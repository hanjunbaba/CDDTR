import numpy as np
import pandas as pd
class CellData:
    def __init__(self, test_fold, cell_types, pert_type, lincs_phase):
        common_perts, all_data_dic = self.prepare_data(cell_types, pert_type, lincs_phase)
        self.cell_types = cell_types
        self.common_perts = common_perts
        if test_fold == 'eval':
            self.train_data_dic = all_data_dic
            self.test_data_dic = all_data_dic
        else:
            train_perts, test_perts, train_data_dic, test_data_dic = \
                self.split_data(cell_types, common_perts, all_data_dic, test_fold, lincs_phase, pert_type)
            self.all_data_dic = all_data_dic
            self.train_perts = train_perts
            self.test_perts = test_perts
            self.train_data_dic = train_data_dic
            self.test_data_dic = test_data_dic


    def load_data(self, cell, pert_type, lincs_phase):
        df = pd.read_csv('./data' + '/' + pert_type + '_' + lincs_phase + '/' +
                         cell + '_' + pert_type + '_LINCS_' + lincs_phase + '.tsv', sep='\t')
        df = df.groupby(['pert_id'], as_index=False).mean()
        pert_ids = df['pert_id'].values
        df = df.drop(['pert_id', 'Unnamed: 0'], axis=1)
        pert_data = df.values
        max_data = np.max(np.abs(np.asarray(pert_data)))
        pert_data = pert_data / max_data
        id_data_dic = {}
        for l_id in range(len(pert_ids)):
            id_data_dic[pert_ids[l_id]] = pert_data[l_id]
        return id_data_dic

    def load_pert_ids(self, cell, pert_type, lincs_phase):
        df = pd.read_csv('./data' + '/' + pert_type + '_' + lincs_phase + '/' +
                         cell + '_' + pert_type + '_LINCS_' + lincs_phase + '.tsv', sep='\t')
        df = df.groupby(['pert_id'], as_index=False).mean()
        pert_ids = df['pert_id'].values
        return pert_ids

    def prepare_data(self, cell_types, pert_type, lincs_phase):
        common_perts = self.load_pert_ids(cell_types[0], pert_type, lincs_phase)
        for cell in cell_types:
            pert_ids = self.load_pert_ids(cell, pert_type, lincs_phase)
            print('cell type: ' + cell)
            print('perts num: ' + str(len(pert_ids)))
            print('pert type: ' + pert_type)
            print('from:LINCS ' + lincs_phase)
            print('##################')
            common_perts = [pert for pert in pert_ids if pert in common_perts]
        # common_perts = np.load('./data/trt_cp_2/all_common.npy')
        print('all common perts: ' + str(len(common_perts)))
        print('#########################')
        all_data_dic = {}
        for cell in cell_types:
            id_data_dic = self.load_data(cell, pert_type, lincs_phase)
            pert_data = []
            for pert in common_perts:
                pert_data.append(id_data_dic[pert])
            all_data_dic[cell] = np.asarray(pert_data)
        return common_perts, all_data_dic

    def split_data(self, cell_types, common_perts, all_data_dic, test_fold, LINCS, pert_type):
        str(test_fold)
        if os.path.exists('./folds/' + str(pert_type) + '/' + str(LINCS) + '/' + str(test_fold)):
            test_perts = np.loadtxt('./folds/' + str(pert_type) + '/' + str(LINCS) + '/' + str(test_fold), dtype='str')
        else:
            test_perts = []

        test_perts = [pert for pert in common_perts if pert in test_perts]
        test_perts_ = [i for i, pert in enumerate(common_perts) if pert in test_perts]

        train_perts = list(set(common_perts) - set(test_perts))
        train_perts_ = [i for i, pert in enumerate(common_perts) if pert in train_perts]

        train_data_dic = {}
        test_data_dic = {}
        for cell in cell_types:
            train_data_dic[cell] = np.asarray(all_data_dic[cell][train_perts_])
            test_data_dic[cell] = np.asarray(all_data_dic[cell][test_perts_])
        print('fold: ' + str(test_fold))
        print('cell nums: ' + str(len(cell_types)))
        print('train set size: ' + str(train_data_dic[cell_types[0]].shape[0]))
        print('test set size: ' + str(test_data_dic[cell_types[0]].shape[0]))
        print('###########################')

        return train_perts, test_perts, train_data_dic, test_data_dic


# def Muti_data(cell_types, pert_types, lincs_phases, test_fold):
#     global CellData
#     pert_data = {}
#     for i in range(len(pert_types)):
#         pert_data[i] = CellData('All_train', cell_types, pert_types[i], lincs_phases[i])
#     common_perts = []
#     for i in range(len(pert_types)):
#         for perts in pert_data[i].common_perts:
#             common_perts.append(perts)
#     common_perts = list(set(common_perts))
#     print('all perts num: '+ str(len(common_perts)))
#     print('#########################')
#     if os.path.exists('./data/test_folds/' + str(test_fold)):
#         test_perts_ = np.loadtxt('./data/test_folds/' + str(test_fold), dtype='str')
#     else:
#         test_perts_ = []
#     train_val_perts = list(set(common_perts) - set(test_perts_))
#     shuffle(train_val_perts)
#     split = int(0.95 * len(train_val_perts))
#     train_perts_ = train_val_perts[:split]
#     val_perts_ = train_val_perts[split:]
#
#     train_data_dic = {}
#     val_data_dic = {}
#     test_data_dic = {}
#     for cell in cell_types:
#         train_data = []
#         val_data = []
#         test_data = []
#         train_perts = []
#         val_perts = []
#         test_perts = []
#         shared_perts = 0
#         for i in range(len(pert_types)):
#             for n, pert in enumerate(pert_data[i].common_perts):
#                 if pert in train_perts or pert in val_perts or pert in test_perts:
#                     shared_perts += 1
#                 if pert in train_perts_ and pert not in train_perts:
#                     train_data.append(pert_data[i].train_data_dic[cell][n])
#                     train_perts.append(pert)
#                 if pert in val_perts_ and pert not in val_perts:
#                     val_data.append(pert_data[i].train_data_dic[cell][n])
#                     val_perts.append(pert)
#                 if pert in test_perts_ and pert not in test_perts:
#                     test_data.append(pert_data[i].train_data_dic[cell][n])
#                     test_perts.append(pert)
#
#         train_data_dic[cell] = np.asarray(train_data)
#         val_data_dic[cell] = np.asarray(val_data)
#         test_data_dic[cell] = np.asarray(test_data)
#     print('cell nums: ' + str(len(cell_types)))
#     print('train set size: ' + str(len(train_perts)))
#     print('validation set size: ' + str(len(val_perts)))
#     print('test set size: ' + str(len(test_perts)))
#     print('shared perts num: ' + str(shared_perts))
#     print('#########################')
#     return common_perts, train_perts, val_perts, test_perts, train_data_dic, val_data_dic, test_data_dic


# CellData = CellData(test_fold='', cell_types=['PC3', 'MCF7', 'HT29'], pert_type='trt_cp', lincs_phase='1')

# common_perts, train_perts, val_perts, test_perts, train_data_dic, val_data_dic, test_data_dic =\
#     Muti_data(cell_types=['PC3', 'MCF7'], pert_types=['trt_sh','trt_cp'], lincs_phases=['1','2'], test_fold='30percent')
# print(CellData.train_data_dic['PC3'].shape)
# print(CellData.test_data_dic['PC3'].shape)
# print(CellData.val_data_dic['PC3'].shape)



