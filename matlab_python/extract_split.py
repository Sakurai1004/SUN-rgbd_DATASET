import scipy.io
import os

# データの読み込み
split_data = scipy.io.loadmat('../OFFICIAL_SUNRGBD/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')

# ハッシュマップの作成
hash_train = []
hash_val = []
N_train = split_data['alltrain'].shape[1]
N_val = split_data['alltest'].shape[1]

for i in range(N_train):
    folder_path = split_data['alltrain'][0][i][0]
    folder_path = "./OFFICIAL_SUNRGBD" + folder_path[16:]
    hash_train.append(folder_path)
for i in range(N_val):
    folder_path = split_data['alltest'][0][i][0]
    folder_path = "./OFFICIAL_SUNRGBD" + folder_path[16:]
    hash_val.append(folder_path)

# データをtrainまたはvalセットにマッピングする
meta_data = scipy.io.loadmat('../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat')
fid_train = open('../sunrgbd_trainval/train_data_idx.txt', 'w')
fid_val = open('../sunrgbd_trainval/val_data_idx.txt', 'w')

for imageId in range(10335):
    data = meta_data['SUNRGBDMeta'][0][imageId]
    depthpath = data['depthpath'][0]
    depthpath = "./OFFICIAL_SUNRGBD" + depthpath[16:]
    filepath = os.path.dirname(os.path.dirname(depthpath))
    if filepath in hash_train:
        fid_train.write(str(imageId) + '\n')
    elif filepath in hash_val:
        fid_val.write(str(imageId) + '\n')
    else:
        a = 1

fid_train.close()
fid_val.close()
