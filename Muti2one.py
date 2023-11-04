import numpy as np
from scipy.stats import pearsonr
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
from DATA import CellData
from Models import con_model, pre_model

cell_list = ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'YAPC']
pred_cell = 'PC3'
pert_type = 'trt_cp'
weight_dir = './model_weights_1570/LINCS_2/'
folds = 10
input_data_dic = {}
for fold in range(1,11):
    input_datas = []
    input_data_dic = {}
    for cell in cell_list:
        input_data_dic[cell] = np.load(
            weight_dir + cell + '_to_' + pred_cell + '/test/' + str(fold) + '/test_inputs.npy')
    for cell in cell_list:
        pred_model = pre_model()
        pred_model.load_weights(weight_dir + cell + '_to_' + pred_cell + '/test/' + str(fold) + '/test_model.h5')
        pred_data = np.asarray(pred_model.predict(input_data_dic[cell]))
        input_datas.append(pred_data)
    # print(np.array(input_datas).shape)
    # a = np.mean(np.array(input_datas),axis=0).shape
    # print(a)
    mean_inputs = np.mean(np.array(input_datas), axis=0)
    output_data = np.load(
        weight_dir + cell_list[0] + '_to_' + pred_cell + '/test/' + str(fold) + '/test_outputs.npy')
    pcc = 0
    for i, pre in enumerate(mean_inputs):
        pcc += pearsonr(output_data[i], pre)[0]
    pcc = pcc/len(list(mean_inputs))
    print(pcc)



