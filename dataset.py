from torch.utils.data import Dataset, DataLoader
import numpy as np
import scanpy as sc


class CSDataSet(Dataset):
    def __init__(self, ctrl, stim):
        '''
        Build dataset of control and stimulated adata
        :param ctrl: control adata
        :param stim: stimulated adata
        '''
        self.ctrl = ctrl.to_df().values
        self.stim = stim.to_df().values

    def __getitem__(self, index):
        x = self.ctrl[index, :]
        y = self.stim[index, :]
        return x, y

    def __len__(self):
        return self.ctrl.shape[0]


class AnnDataSet(Dataset):
    def __init__(self, adata):
        '''
        Build dataset of adata
        :param adata: adata of training or testing set
        '''
        self.data = adata.to_df().values

    def __getitem__(self, index):
        x = self.data[index, :]
        return x

    def __len__(self):
        return self.data.shape[0]


def data_loader(adata, key_dic, bz=128, cell_to_pred='CD4T'):
    '''
    Build the training set from the input adata
    :param adata: complete adata
    :param key_dic: dictionary of keywords for the data set
    :param bz: batch size
    :param cell_to_pred: cell type to predict
    :return: training set
    '''
    cell_type_key = key_dic['cell_type_key']
    condition_key = key_dic['condition_key']
    ctrl_key = key_dic['ctrl_key']
    stim_key = key_dic['stim_key']
    types = list(set(adata.obs[cell_type_key]))
    new_adata = adata[(adata.obs[cell_type_key] == 'none')]
    for cell_type in types:
        if cell_type != cell_to_pred:
            ctrl = adata[((adata.obs[cell_type_key] == cell_type) &
                          (adata.obs[condition_key] == ctrl_key))]
            stim = adata[((adata.obs[cell_type_key] == cell_type) &
                          (adata.obs[condition_key] == stim_key))]

            ctrl_ind = np.random.choice(range(ctrl.shape[0]), size=(int(ctrl.shape[0] / bz) + 1) * bz)
            stim_ind = np.random.choice(range(stim.shape[0]), size=(int(stim.shape[0] / bz) + 1) * bz)
            ctrl_adata = ctrl[ctrl_ind, :]
            stim_adata = stim[stim_ind, :]
            ctrl_adata.obs["label"] = [cell_type + '_ctrl'] * ctrl_adata.obs.shape[0]
            stim_adata.obs["label"] = [cell_type + '_stim'] * stim_adata.obs.shape[0]
            if new_adata.obs.shape[0] == 0:
                new_adata = ctrl_adata.concatenate(stim_adata)
            else:
                new_adata = new_adata.concatenate(ctrl_adata, stim_adata)
        else:
            ctrl = adata[((adata.obs[cell_type_key] == cell_type) &
                          (adata.obs[condition_key] == ctrl_key))]
            ctrl_ind = np.random.choice(range(ctrl.shape[0]), size=(int(ctrl.shape[0] / bz) + 1) * bz)
            ctrl_adata = ctrl[ctrl_ind, :]
            ctrl_adata.obs["label"] = [cell_type + '_ctrl'] * ctrl_adata.obs.shape[0]
            if new_adata.obs.shape[0] == 0:
                new_adata = ctrl_adata
            else:
                new_adata = new_adata.concatenate(ctrl_adata)
    train_set = AnnDataSet(new_adata)
    train_loader = DataLoader(dataset=train_set, batch_size=bz, shuffle=False,
                              num_workers=0, drop_last=False)
    return train_loader


if __name__ == '__main__':
    adata = sc.read_h5ad('../data/pbmc_perturbed.h5ad')
    input_dim = adata.var.shape[0]
    cell_to_pred = 'CD4T'
    key_dic = {'condition_key': 'condition',
               'cell_type_key': 'cell_type',
               'ctrl_key': 'control',
               'stim_key': 'stimulated'}

    train_loader = data_loader(adata, key_dic, bz=128, cell_to_pred='CD4T')