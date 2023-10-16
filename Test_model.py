import numpy as np
from scipy.stats import pearsonr
from Models import pre_model
from sklearn.metrics import mean_squared_error
from Train_model import setDir, get_options


def test(cell_list, fold, LINCS, result_dir, model_dir, result_txt):
    print('cells: ' + cell_list[0] + ' to ' + cell_list[1])
    print('test fold:' + str(fold))
    setDir(result_dir)
    with open(result_dir + '/' + result_txt + '.txt', "a") as f:
        f.write('\n')
        f.write(cell_list[0] + ' to ' + cell_list[1])
        f.write('\n')
        f.close()
    test_m = pre_model()
    model_dir = model_dir + '/LINCS_' + LINCS + '/' + cell_list[0] + '_to_' + cell_list[1] + '/test/' + str(fold)
    test_m.load_weights(model_dir + '/test_model.h5')
    test_inputs = np.load(model_dir + '/test_inputs.npy')
    # np.save('./test_np/'+cell_list[0]+'/'+str(fold), test_inputs)
    # np.save('./test_np/pre_' + cell_list[1] + '_Baseline/' + str(fold), test_inputs)
    test_outputs = np.load(model_dir + '/test_outputs.npy')
    pre = np.asarray(test_m.predict(test_inputs))
    # np.save('./test_np/pre_' + cell_list[1] + '_CDDTR/' + str(fold), pre)
    print('Number of test sets:' + str(len(pre)))
    pcc_base = 0
    pcc_fold = 0
    for num in range(len(pre)):
        pcc_fold += pearsonr(test_outputs[num], pre[num])[0]
        pcc_base += pearsonr(test_outputs[num], test_inputs[num])[0]
    mse = mean_squared_error(test_outputs, pre)
    rmse = np.sqrt(mse)
    with open(result_dir + '/' + result_txt + '.txt', "a") as f:
        f.write('######## fold '+str(fold)+' #########')
        f.write('\n')
        f.write('pcc(Baseline):' + str(pcc_base / len(pre)))
        f.write('\n')
        f.write('pcc(Ours):' + str(pcc_fold / len(pre)))
        f.write('\n')
        f.write('mse:' + str(mse))
        f.write('\n')
        f.write('rmse:' + str(rmse))
        f.write('\n')
        f.close()

if __name__ == '__main__':
    opt = get_options()
    cells = opt.cells
    if opt.fold == 0:
        folds = range(1, 11)
    elif opt.fold == 1:
        folds = ['30percent']
    LINCS = opt.lincs_phase
    for cell_1 in cells:
        for cell_2 in cells:
            if cell_1 != cell_2:
                for fold in folds:
                    cell_list = [cell_1, cell_2]
                    test(cell_list, str(fold), LINCS, opt.save_dir_r, opt.save_dir_m, opt.result_txt)