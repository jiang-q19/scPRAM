import os
import numpy as np
import scanpy as sc


def get_less_adata(adata, key_dic, frac=0.2):
    """
    Dataset downsampling
    :param adata: input adata
    :param key_dic: dictionary of keywords for the dataset, such as
               {'condition_key': 'condition',
               'cell_type_key': 'cell_type',
               'ctrl_key': 'control',
               'stim_key': 'stimulated',
               'pred_key': 'pred',
               }
    :param frac: downsampling rate
    :return: New dataset obtained by downsampling
    """
    if (frac > 1) | (frac < 0):
        print("Info: the fac should be in [0,1]")
        return None
    types = set(adata.obs[key_dic['cell_type_key']])
    conditions = set(adata.obs[key_dic['condition_key']])
    res = {}
    for i in range(int(1 / (frac + 0.001)) + 1):
        res['part{}'.format(i)] = []
    for cell_to_pred in types:
        for condition in conditions:
            adata_new = adata[((adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                               (adata.obs[key_dic['condition_key']] == condition))]
            size = int(frac * adata_new.n_obs)
            idxs = np.array(range(adata_new.n_obs))
            np.random.shuffle(idxs)
            for i in range(int(1 / (frac + 0.001))):
                idx = idxs[0:size * (i + 1)]
                res['part{}'.format(i)].append(adata_new[idx])
            res['part{}'.format(int(1 / (frac + 0.001)))].append(adata_new)
    for i in range(int(1 / (frac + 0.001)) + 1):
        res['part{}'.format(i)] = sc.AnnData.concatenate(*res['part{}'.format(i)], join='outer')
    return res


def check_dir(path='./'):
    """
    Check that project folders exist for the specified directory, and create them if not
    :param path: the specified directory
    :return: None
    """
    img_path = path + 'images'
    log_path = path + 'log'
    res_path = path + 'results'
    mod_path = path + "save_models"

    if not os.path.isdir(img_path):
        os.mkdir(img_path)
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    if not os.path.isdir(mod_path):
        os.mkdir(mod_path)


# 用于平衡各个类型的数目, 按照最多进行补齐或者最少进行采样
def balancer(adata, type_key, max=True):
    """
    Balance the number of cells of each type, with maximum replenishment or minimum sampling
    :param adata: the input adata
    :param type_key: the key of cell type in adata.obs
    :param max: maximum if True and minimum if False
    :return: the balanced data
    """
    class_names = np.unique(adata.obs[type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata[adata.obs[type_key] == cls].shape[0]
    if max:
        number = np.max(list(class_pop.values()))
    else:
        number = np.min(list(class_pop.values()))
    index_all = []
    for cls in class_names:
        class_index = np.array(adata.obs[type_key] == cls)
        index_cls = np.nonzero(class_index)[0]
        index_cls_r = index_cls[np.random.choice(len(index_cls), number)]
        index_all.append(index_cls_r)

    balanced_data = adata[np.concatenate(index_all)].copy()
    return balanced_data


def get_ctrl_stim(latent_adata, key_dic):
    """
    Get the control and perturbed latent data after balance
    :param latent_adata: the input latent adata
    :param key_dic: dictionary of keywords for the dataset, such as
               {'condition_key': 'condition',
               'cell_type_key': 'cell_type',
               'ctrl_key': 'control',
               'stim_key': 'stimulated',
               'pred_key': 'pred',
               }
    :return: the control and perturbed latent data after balance
    """
    condition_key = key_dic['condition_key']
    cell_type_key = key_dic['cell_type_key']
    ctrl_key = key_dic['ctrl_key']
    stim_key = key_dic['stim_key']
    ctrl_x = latent_adata[latent_adata.obs[condition_key] == ctrl_key, :]
    stim_x = latent_adata[latent_adata.obs[condition_key] == stim_key, :]
    ctrl_x = balancer(ctrl_x, cell_type_key)
    stim_x = balancer(stim_x, cell_type_key)
    eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
    cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
    stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
    ctrl_z = ctrl_x[cd_ind, :]
    stim_z = stim_x[stim_ind, :]
    return ctrl_z, stim_z


# 获取对照组的数据
def get_ctrl_adata(adata, cell_to_pred, key_dic):
    """
    Get the control adata of certain cell type
    :param adata: the input adata
    :param cell_to_pred: the cell type that you want to predict
    :param key_dic: key_dic: dictionary of keywords for the dataset, such as
               {'condition_key': 'condition',
               'cell_type_key': 'cell_type',
               'ctrl_key': 'control',
               'stim_key': 'stimulated',
               'pred_key': 'pred',
               }
    :return: control adata
    """
    ctrl_adata = adata[((adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                        (adata.obs[key_dic['condition_key']] == key_dic['ctrl_key']))]
    return ctrl_adata



