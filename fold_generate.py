from sklearn.model_selection import KFold
from DATA import CellData

def Split_Sets_Fold(total_fold, data):
    train_index = []
    test_index = []
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=True)
    # 这里设置shuffle设置为ture就是打乱顺序在分配
    for train_i, test_i in kf.split(data):
        train_index.append(train_i)
        test_index.append(test_i)
    return train_index, test_index

CellData = CellData(test_fold='', cell_types=['A375', 'PC3'], pert_type='trt_oe', lincs_phase='1')
pert_ids = CellData.common_perts
train_index, test_index = Split_Sets_Fold(10, pert_ids)

for fold in range(1, 11):
    with open(str(fold),'w') as f:
        for i in test_index[fold-1]:
            f.write(pert_ids[i])
            f.write('\n')
        f.close()
# random.shuffle(pert_ids)
# l = int(len(pert_ids)/10*3)
# test_pert = pert_ids[:l]
# print(len(test_pert))
# with open('2_30percent','w') as f:
#     for n in test_pert:
#         f.write(n)
#         f.write('\n')
#     f.close()
