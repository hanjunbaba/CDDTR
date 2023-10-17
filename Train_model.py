import os
import shutil
from DATA import CellData
from Models import con_model, pre_model
import numpy as np
from tensorflow import keras
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
import argparse

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cells', default=['A375', 'PC3'],
                        type=list, help='Train between pairs of cell lines stored in the list')
    parser.add_argument('-pert', default='trt_cp',
                        type=str, help='Perturbation type')
    parser.add_argument('-fold', default=0,
                        type=int, help='set 0 to start ten fold cross validation'
                                       '/ 1 to 70 percent training'
                                       '/ 2 to start training all')
    parser.add_argument('-p_epochs', default=1000,
                        type=int, help='Epochs of training'
                        )
    parser.add_argument('-epochs', default=1000,
                        type=int, help='Epochs of pre training'
                        )
    parser.add_argument('-p_lr', default=1e-4,
                        type=float, help='Learning rate of pre training'
                        )
    parser.add_argument('-lr', default=2e-4,
                        type=float, help='Learning rate of training'
                        )
    parser.add_argument('-p_batch_size', default=128,
                        type=int, help='Batch size of pre training'
                        )
    parser.add_argument('-batch_size', default=128,
                        type=int, help='Batch size of training'
                        )
    parser.add_argument('-input_dropout_rate', default=0.1,
                        type=float, help='Dropout rate of inputs'
                        )
    parser.add_argument('-lincs_phase', default='2',
                        type=str, help='Lincs phase'
                        )
    parser.add_argument('-save_dir_m', default='./model_weights',
                        type=str, help='The folder where the model parameters are stored'
                        )
    parser.add_argument('-save_dir_r', default='./result',
                        type=str, help='The folder where the test results are stored'
                        )
    parser.add_argument('-pre_train', default=False,
                        type=bool, help='Whether to use pre training'
                        )
    parser.add_argument('-result_txt', default='result',
                        type=str, help='The file name where the results are saved as txt, txt is saved in save_dir_r'
                        )

    opt = parser.parse_args()

    return opt

def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)

def setDir_rm(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath, ignore_errors=True)
        os.mkdir(filepath)
def pre_train(cell_data, LINCS, model_dir, opt):
    print('Start train model ' + cell_data.cell_types[0] + ' to ' + cell_data.cell_types[1])
    print('start train pre-model')
    pre_m = pre_model()
    pre_m.compile(loss=mean_squared_error, optimizer=Adam(lr=opt.p_lr))
    pre_m.fit(cell_data.train_data_dic[cell_data.cell_types[0]], cell_data.train_data_dic[cell_data.cell_types[0]],
              shuffle=True,
              epochs=opt.p_epochs,
              batch_size=opt.p_batch_size,
              )
    setDir(model_dir)
    model_dir = model_dir + '/LINCS_' + LINCS
    setDir(model_dir)
    model_dir = model_dir + '/' + cell_data.cell_types[0] + '_to_' + cell_data.cell_types[1]
    setDir(model_dir)
    setDir_rm(model_dir + '/pre-train')
    pre_m.save_weights(model_dir + '/pre-train/' + 'pre_model.h5', save_format="h5")

    pre_m.get_layer('en').save_weights(model_dir + '/pre-train/' + 'en.h5', save_format="h5")
    pre_m.get_layer('de').save_weights(model_dir + '/pre-train/' + 'de.h5', save_format="h5")
    print('AE has been saved')

def train_model(cell_data, fold, LINCS, model_dir, opt):
    print('start train con-model')
    f = str(fold)
    print('+++++++++++ fold '+str(f)+' +++++++++++')
    con_m = con_model()
    model_dir_ = model_dir
    setDir(model_dir)
    model_dir = model_dir + '/LINCS_' + LINCS
    setDir(model_dir)
    model_dir = model_dir + '/' + cell_data.cell_types[0] + '_to_' + cell_data.cell_types[1]
    setDir(model_dir)
    if opt.pre_train:
        model_dir = model_dir
        con_m.get_layer('en_A').load_weights(model_dir + '/pre-train/' + 'en.h5')
        con_m.get_layer('de_A').load_weights(model_dir + '/pre-train/' + 'de.h5')
    con_m.compile(loss=mean_squared_error, optimizer=Adam(lr=opt.lr))
    con_m.fit([cell_data.train_data_dic[cell_data.cell_types[0]], cell_data.train_data_dic[cell_data.cell_types[1]]],
              [cell_data.train_data_dic[cell_data.cell_types[0]], cell_data.train_data_dic[cell_data.cell_types[1]],
               cell_data.train_data_dic[cell_data.cell_types[1]], cell_data.train_data_dic[cell_data.cell_types[0]]],
              shuffle=True,
              epochs=opt.epochs,
              batch_size=opt.batch_size,
              )
    setDir(model_dir + '/con-train')
    setDir_rm(model_dir + '/con-train/' + str(f))
    con_m.save_weights(model_dir + '/con-train/' + str(f) + '/con_model.h5', save_format="h5")

    con_m.get_layer('en_A').save_weights(model_dir + '/con-train/' + str(f) + '/en_A.h5', save_format="h5")
    con_m.get_layer('de_B').save_weights(model_dir + '/con-train/' + str(f) + '/de_B.h5', save_format="h5")
    test_m = pre_model()
    test_m.get_layer('en').load_weights(model_dir + '/con-train/' + str(f) + '/en_A.h5')
    test_m.get_layer('de').load_weights(model_dir + '/con-train/' + str(f) + '/de_B.h5')
    setDir(model_dir + '/test')
    setDir_rm(model_dir + '/test/' + str(f))
    test_m.save_weights(model_dir + '/test/' + str(f) + '/test_model.h5', save_format="h5")
    test_inputs = cell_data.test_data_dic[cell_data.cell_types[0]]
    test_outputs = cell_data.test_data_dic[cell_data.cell_types[1]]
    np.save(model_dir + '/test/' + str(f) + '/test_inputs.npy', test_inputs)
    np.save(model_dir + '/test/' + str(f) + '/test_outputs.npy', test_outputs)

    if not opt.pre_train:
        setDir(model_dir_)
        model_dir = model_dir_ + '/LINCS_' + LINCS
        setDir(model_dir)
        model_dir = model_dir + '/' + cell_data.cell_types[1] + '_to_' + cell_data.cell_types[0]
        setDir(model_dir)
        setDir(model_dir + '/con-train')
        setDir_rm(model_dir + '/con-train/' + str(f))
        con_m.save_weights(model_dir + '/con-train/' + str(f) + '/con_model.h5', save_format="h5")
        con_m.get_layer('en_B').save_weights(model_dir + '/con-train/' + str(f) + '/en_B.h5', save_format="h5")
        con_m.get_layer('de_A').save_weights(model_dir + '/con-train/' + str(f) + '/de_A.h5', save_format="h5")
        test_m = pre_model()
        test_m.get_layer('en').load_weights(model_dir + '/con-train/' + str(f) + '/en_B.h5')
        test_m.get_layer('de').load_weights(model_dir + '/con-train/' + str(f) + '/de_A.h5')
        setDir(model_dir + '/test')
        setDir_rm(model_dir + '/test/' + str(f))
        test_m.save_weights(model_dir + '/test/' + str(f) + '/test_model.h5', save_format="h5")
        test_inputs = cell_data.test_data_dic[cell_data.cell_types[1]]
        test_outputs = cell_data.test_data_dic[cell_data.cell_types[0]]
        np.save(model_dir + '/test/' + str(f) + '/test_inputs.npy', test_inputs)
        np.save(model_dir + '/test/' + str(f) + '/test_outputs.npy', test_outputs)
    print('Save model completed: fold ' + str(f))
    keras.backend.clear_session()


if __name__ == '__main__':
    opt = get_options()
    print(opt)
    cells = opt.cells
    if opt.fold == 0:
        folds = range(1, 11)
    elif opt.fold == 1:
        folds = ['30percent']
    elif opt.fold == 1:
        folds = ['eval']
    print([fold for fold in folds])
    pre_tr = False

    l = len(cells)
    if not opt.pre_train:
        for p in range(l-1):
            for q in range(p+1, l):
                cell_list = [cells[p], cells[q]]
                for fold in folds:
                    cell_data = CellData(fold, cell_list, opt.pert, opt.lincs_phase)
                    train_model(cell_data, str(fold), opt.lincs_phase, opt.save_dir_m, opt)
    if opt.pre_train:
        for p in range(l):
            for q in range(l):
                if p != q:
                    cell_list = [cells[p], cells[q]]
                    for fold in folds:
                        cell_data_ = CellData('eval', cell_list, opt.pert, opt.lincs_phase)
                        pre_train(cell_data_, opt.lincs_phase, opt.save_dir_m, opt)
                        cell_data = CellData(fold, cell_list, opt.pert, opt.lincs_phase)
                        train_model(cell_data, str(fold), opt.lincs_phase, opt.save_dir_m, opt)

