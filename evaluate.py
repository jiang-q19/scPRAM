import seaborn as sns
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.stats import wasserstein_distance
from matplotlib.backends.backend_pdf import PdfPages
import sklearn
import ot


def evaluate_adata(eval_adata, cell_type, key_dic):
    """
    Evaluate predictive performance; Note that this function is used when one of the subexperiments
    on a dataset have been completed
    :param cell_type: the cell type that you are evaluating
    :param eval_adata: adata of gene expression containing ctrl,stim, pred
    :param key_dic: dictionary of keywords for the dataset, such as
               {'condition_key': 'condition',
               'cell_type_key': 'cell_type',
               'ctrl_key': 'control',
               'stim_key': 'stimulated',
               'pred_key': 'pred',
               }
    :return: None
    """
    # PCA cluster results
    sc.tl.pca(eval_adata)
    sc.pl.pca(eval_adata, color=key_dic['condition_key'], frameon=False,
              title="PCA of " + cell_type + " by Condition")
    # conmmon degs of top 100
    sc.tl.rank_genes_groups(eval_adata, groupby=key_dic['condition_key'],
                            reference=key_dic['ctrl_key'], method="wilcoxon")
    degs_pred = eval_adata.uns["rank_genes_groups"]["names"][key_dic['pred_key']]
    degs_ctrl = eval_adata.uns["rank_genes_groups"]["names"][key_dic['stim_key']]
    common_degs = list(set(degs_ctrl[0:100]) & set(degs_pred[0:100]))
    common_nums = len(common_degs)
    print("common DEGs: ", common_nums)
    # regression plot of mean
    draw_reg_plot(eval_adata=eval_adata,
                  cell_type=cell_type,
                  reg_type='mean',
                  axis_keys={"x": "pred", "y": key_dic['stim_key']},
                  condition_key=key_dic['condition_key'],
                  gene_draw=degs_ctrl[:10],
                  top_gene_list=degs_ctrl[:100],
                  save_path=None,
                  title=None,
                  show=True,
                  return_fig=False,
                  fontsize=12
                  )
    # regression plot of var
    draw_reg_plot(eval_adata=eval_adata,
                  cell_type=cell_type,
                  reg_type='var',
                  axis_keys={"x": "pred", "y": key_dic['stim_key']},
                  condition_key=key_dic['condition_key'],
                  gene_draw=degs_ctrl[:10],
                  top_gene_list=degs_ctrl[:100],
                  save_path=None,
                  title=None,
                  show=True,
                  return_fig=False,
                  fontsize=12
                  )
    # violin contrast
    gene = degs_ctrl[0]
    sc.pl.violin(eval_adata, keys=gene, groupby=key_dic['condition_key'])
    gene = degs_ctrl[1]
    sc.pl.violin(eval_adata, keys=gene, groupby=key_dic['condition_key'])
    gene = degs_ctrl[2]
    sc.pl.violin(eval_adata, keys=gene, groupby=key_dic['condition_key'])
    # degs contrast
    sc.tl.rank_genes_groups(eval_adata, groupby=key_dic['condition_key'],
                            reference=key_dic['ctrl_key'], method="wilcoxon")
    sc.pl.rank_genes_groups(eval_adata, n_genes=25, sharey=False, show=True)

    marker_genes = degs_ctrl[0:20]
    sc.pl.dotplot(eval_adata, marker_genes, groupby=key_dic['condition_key'], show=True)


def evaluate(data_name, eval_path, key_dic, save_path=None, return_fig=False):
    '''
    Evaluate predictive performance; Note that this function is used when all the subexperiments
    on a dataset have been completed
    :param data_name: name of dataset, such as PBMC
    :param eval_path: the path where you save the results of prediction
    :param key_dic: dictionary of keywords for the dataset
    :param save_path: the path to save the evaluation results
    :param return_fig: whether to return the result of drawing
    :return: DataFrame of the evaluation indicator
    '''
    adata = sc.read_h5ad(eval_path)
    ctrl_key = key_dic['ctrl_key']
    stim_key = key_dic['stim_key']
    pred_key = key_dic['pred_key']
    cell_type_key = key_dic['cell_type_key']
    condition_key = key_dic['condition_key']
    types = list(set(adata.obs[cell_type_key]))
    types.sort()
    degs_list = []
    r2_mean_list = []
    r2_std_list = []
    fig_list = []
    dist_list = []
    for cell_type in types:
        print(cell_type)
        # 取出某一类型的数据
        eval_adata = adata[(adata.obs[cell_type_key] == cell_type)]
        # 决定系数
        r2_mean, r2_std = get_pearson2(eval_adata, key_dic)
        r2_mean_list.append(r2_mean.values.tolist())
        r2_std_list.append(r2_std.values.tolist())
        # PCA对比
        sc.tl.pca(eval_adata)
        fig = sc.pl.pca(eval_adata, color=condition_key, frameon=False, show=False,
                        title="PCA of " + cell_type + " by Condition", return_fig=True)
        fig_list.append(fig)
        # DEGs个数
        sc.tl.rank_genes_groups(eval_adata, groupby=condition_key, reference=ctrl_key, method="wilcoxon")
        degs_pred = eval_adata.uns["rank_genes_groups"]["names"][pred_key]
        degs_ctrl = eval_adata.uns["rank_genes_groups"]["names"][stim_key]
        common_degs = list(set(degs_ctrl[0:100]) & set(degs_pred[0:100]))
        common_nums = len(common_degs)
        degs_list.append(common_nums)
        print(common_nums)
        # 线性回归
        r2mean = draw_reg_plot(eval_adata=eval_adata,
                               cell_type=cell_type,
                               reg_type='mean',
                               axis_keys={"x": pred_key, "y": stim_key},
                               condition_key=condition_key,
                               gene_draw=degs_ctrl[:10],
                               top_gene_list=degs_ctrl[:100],
                               save_path=None,
                               title=None,
                               show=False,
                               return_fig=True,
                               fontsize=12
                               )
        fig_list.append(r2mean[2])

        r2var = draw_reg_plot(eval_adata=eval_adata,
                              cell_type=cell_type,
                              reg_type='var',
                              axis_keys={"x": pred_key, "y": stim_key},
                              condition_key=condition_key,
                              gene_draw=degs_ctrl[:10],
                              top_gene_list=degs_ctrl[:100],
                              save_path=None,
                              title=None,
                              show=False,
                              return_fig=True,
                              fontsize=12
                              )
        fig_list.append(r2var[2])
        # Violin对比: 选取前3的基因作为对比
        for i in range(3):
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            gene = degs_ctrl[i]

            sc.pl.violin(eval_adata, keys=gene, groupby=condition_key, ax=ax, show=False, )
            ax.set_title('Violin Plot of {} in {}'.format(gene, cell_type))
            fig_list.append(fig)
        # Wasserstein距离计算
        dist = get_wasserstein_distance(eval_adata, case_key=stim_key, pred_key=pred_key, top_genes=degs_ctrl[:100])
        dist_list.append(dist)
    # 将定量指标改为dataframe
    data = np.array([degs_list,
                     [i[0] for i in r2_mean_list],
                     [i[2] for i in r2_mean_list],
                     [i[1] for i in r2_mean_list],
                     [i[3] for i in r2_mean_list],
                     [i[0] for i in dist_list],
                     [i[1] for i in dist_list],
                     [i[0] for i in r2_std_list],
                     [i[2] for i in r2_std_list],
                     [i[1] for i in r2_std_list],
                     [i[3] for i in r2_std_list]])
    df = pd.DataFrame(data.T,
                      columns=['common DEGs of top 100 genes',
                               'mean R2 of all genes expression mean', 'mean R2 of top 100 genes expression mean',
                               'mean R2 of all genes expression var', 'mean R2 of top 100 expression var',
                               'Wasserstein distance of all genes', 'Wasserstein distance of top 100 genes',
                               'std R2 of all genes expression mean', 'std R2 of top 100 genes expression mean',
                               'std R2 of all genes expression var', 'std R2 of top 100 expression var'],
                      index=types)
    # 如果提供路径，则保存为对应路径下的csv
    if save_path:
        df.to_csv(save_path + '/log/{}.csv'.format(data_name))
    if return_fig and save_path:
        with PdfPages(save_path + '/images/{}.pdf'.format(data_name)) as pdf:
            for i in range(len(fig_list)):
                pdf.savefig(figure=fig_list[i], dpi=200, bbox_inches='tight')
                plt.close()
    return df


def get_pearson2(eval_adata, key_dic, n_degs=100, sample_ratio=0.8, times=100):
    """
    Calculating the regression coefficient between the predicted perturbation response and the true response
    :param eval_adata: adata of gene expression containing ctrl,stim, pred
    :param key_dic: dictionary of keywords for the dataset, such as
               {'condition_key': 'condition',
               'cell_type_key': 'cell_type',
               'ctrl_key': 'control',
               'stim_key': 'stimulated',
               'pred_key': 'pred',
               }
    :param n_degs: number of differentially expressed genes used
    :param sample_ratio: sampling rate
    :param times: Sampling times
    :return: DataFrame of the square of pearson's coefficient
    """
    stim_key = key_dic['stim_key']
    pred_key = key_dic['pred_key']
    ctrl_key = key_dic['ctrl_key']
    condition_key = key_dic['condition_key']
    sc.tl.rank_genes_groups(eval_adata, groupby=condition_key, reference=ctrl_key, method="wilcoxon")
    degs = eval_adata.uns["rank_genes_groups"]["names"][stim_key][:n_degs]
    df_stim = eval_adata[(eval_adata.obs[condition_key] == stim_key)].to_df()
    df_pred = eval_adata[(eval_adata.obs[condition_key] == pred_key)].to_df()
    data = np.zeros((times, 4))
    for i in range(times):
        stim = df_stim.sample(frac=sample_ratio, random_state=i)
        pred = df_pred.sample(frac=sample_ratio, random_state=i)
        # 全部基因的均值
        stim_mean = stim.mean().values.reshape(1, -1)
        pred_mean = pred.mean().values.reshape(1, -1)
        # 全部基因的方差
        stim_var = stim.var().values.reshape(1, -1)
        pred_var = pred.var().values.reshape(1, -1)
        # 全部基因的均值和方差决定系数
        r2_mean = (np.corrcoef(stim_mean, pred_mean)[0, 1]) ** 2
        r2_var = (np.corrcoef(stim_var, pred_var)[0, 1]) ** 2
        # 只看Top 100 DEGs的基因
        stim_degs_mean = stim.loc[:, degs].mean().values.reshape(1, -1)
        pred_degs_mean = pred.loc[:, degs].mean().values.reshape(1, -1)
        stim_degs_var = stim.loc[:, degs].var().values.reshape(1, -1)
        pred_degs_var = pred.loc[:, degs].var().values.reshape(1, -1)
        r2_degs_mean = (np.corrcoef(stim_degs_mean, pred_degs_mean)[0, 1]) ** 2
        r2_degs_var = (np.corrcoef(stim_degs_var, pred_degs_var)[0, 1]) ** 2
        data[i, :] = [r2_mean, r2_var, r2_degs_mean, r2_degs_var]
    df = pd.DataFrame(data, columns=['r2_all_mean', 'r2_all_var', 'r2_degs_mean', 'r2_degs_var'])
    r2_mean = df.mean(axis=0)
    r2_std = df.std(axis=0)
    return r2_mean, r2_std


def draw_reg_plot(eval_adata,
                  cell_type,
                  reg_type='mean',
                  axis_keys={"x": "pred", "y": "stimulated"},
                  condition_key='condition',
                  gene_draw=None,
                  top_gene_list=None,
                  save_path=None,
                  title=None,
                  show=True,
                  return_fig=False,
                  fontsize=14
                  ):
    '''
    Plot the regression line between the predicted perturbation response and the true response
    :param eval_adata: adata of gene expression containing ctrl,stim, pred
    :param cell_type: cell type of the data
    :param reg_type: mean or var
    :param axis_keys: {"x": "pred", "y": "stimulated"}, maybe you need change the values of keys
    :param condition_key: 'condition' by default
    :param gene_draw: a list of genes which you want to color red
    :param top_gene_list: a list of top DEGs whose regression coefficients are to be calculated
    :param save_path: the path where you save the regression figure
    :param title: the title of the the regression figure
    :param show: whether to show the figure
    :param return_fig: whether to return the figure
    :param fontsize: the fontsize of the figure
    :return:
    '''
    df_case = eval_adata[(eval_adata.obs[condition_key] == axis_keys["y"])].to_df()
    df_pred = eval_adata[(eval_adata.obs[condition_key] == axis_keys["x"])].to_df()
    if reg_type == 'mean':
        mean_case = df_case.mean().values.reshape(-1, 1)
        mean_pred = df_pred.mean().values.reshape(-1, 1)
    elif reg_type == 'var':
        mean_case = df_case.var().values.reshape(-1, 1)
        mean_pred = df_pred.var().values.reshape(-1, 1)
    data = np.hstack((mean_case, mean_pred))
    data_df = pd.DataFrame(data, columns=['case', 'predict'], index=df_case.columns)

    fig, ax = plt.subplots()
    sns.set(color_codes=True)
    sns.regplot(x='case', y='predict', data=data_df, ax=ax)
    if gene_draw is not None:
        texts = []
        x = mean_case
        y = mean_pred
        for i in gene_draw:
            j = eval_adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(plt.text(x_bar, y_bar, i, fontsize=11, color="black"))
            ax.plot(x_bar, y_bar, "o", color="red", markersize=5)
        adjust_text(
            texts,
            x=x,
            y=y,
            ax=ax,
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
            force_points=(0.0, 0.0),
        )
    if top_gene_list is not None:
        data_deg = data_df.loc[top_gene_list, :]
        r_top = round(data_deg['case'].corr(data_deg['predict'], method='pearson'), 3)
        xt = 0.1 * np.max(data_df['case'])
        yt = 0.85 * np.max(data_df['predict'])
        ax.text(xt, yt, s='$R^2_{top 100 genes}$=' + str(round(r_top * r_top, 3)), fontsize=fontsize, color='black')
    r = round(data_df['case'].corr(data_df['predict'], method='pearson'), 3)
    xt = 0.1 * np.max(data_df['case'])
    yt = 0.75 * np.max(data_df['predict'])
    ax.text(xt, yt, s='$R^2_{all genes}$=' + str(round(r * r, 3)), fontsize=fontsize, color='black')
    if title:
        plt.title(title)
    else:
        plt.title('The Linear Regression of True and Predict Expression ' + reg_type + ' of ' + cell_type)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    if show:
        plt.show()
        plt.close()
    if return_fig:
        return [round(r * r, 3), round(r_top * r_top, 3), fig]
    else:
        return [round(r * r, 3), round(r_top * r_top, 3)]


def get_wasserstein_distance(eval_adata, case_key='stimulated', pred_key='pred', 
                             top_genes=None, cal_type='sum'):
    """
    '''
    This function is used to calculate the Wasserstein distance between the predicted response and the real response
    :param eval_adata: adata of gene expression containing ctrl,stim, pred
    :param case_key: key of perturbed condition
    :param pred_key: key of predictive condition
    :param top_genes: a list of top DEGs whose Wasserstein distance are to be calculated
    :param cal_type: 'sum' or 'mean'
    :return: the Wasserstein distance between the predicted response and the real response
    '''
    """
    dist_list = []
    dist_list_top = []
    pred = eval_adata[(eval_adata.obs["condition"] == pred_key)].to_df()
    case = eval_adata[(eval_adata.obs["condition"] == case_key)].to_df()

    for i in range(pred.shape[1]):
        gene_pred = pred.iloc[:, i].values
        gene_case = case.iloc[:, i].values
        dist = wasserstein_distance(gene_pred, gene_case)
        dist_list.append(dist)

    if top_genes is None:
        res = None
        if cal_type == 'mean':
            res = np.mean(dist_list)
        elif cal_type == 'sum':
            res = np.sum(dist_list)
        return res
    else:
        for gene in top_genes:
            gene_pred = pred.loc[:, gene].values
            gene_case = case.loc[:, gene].values
            dist = wasserstein_distance(gene_pred, gene_case)
            dist_list_top.append(dist)
        res = None
        if cal_type == 'mean':
            res = [np.mean(dist_list), np.mean(dist_list_top)]
        elif cal_type == 'sum':
            res = [np.sum(dist_list), np.sum(dist_list_top)]
        return res


def cluster_evaluation(adata_obs, label_key, cluster_key):
    """
    Evaluation of clustering effect
    :param adata_obs: the observation of the adata
    :param label_key: the key of true label in adata_obs
    :param cluster_key: the key of cluster label in adata_obs
    :return: clustering effect evaluation indicators including AMI, ARI, NMI, HOM, completeness, vms, FMS
    """
    print("clustered by:", cluster_key)
    ARI = sklearn.metrics.adjusted_rand_score(adata_obs[cluster_key], adata_obs[label_key])
    AMI = sklearn.metrics.adjusted_mutual_info_score(adata_obs[label_key], adata_obs[cluster_key])
    NMI = sklearn.metrics.normalized_mutual_info_score(adata_obs[cluster_key], adata_obs[label_key])
    HOM = sklearn.metrics.homogeneity_score(adata_obs[label_key], adata_obs[cluster_key])
    completeness = sklearn.metrics.completeness_score(adata_obs[label_key], adata_obs[cluster_key])
    vms = sklearn.metrics.v_measure_score(adata_obs[label_key], adata_obs[cluster_key])
    FMS = sklearn.metrics.fowlkes_mallows_score(adata_obs[label_key], adata_obs[cluster_key])

    print('AMI:%.3f\tARI:%.3f\tNMI:%.3f\tHOM:%.3f\tcompleteness:%.3f\tV-measure:%.3f\tFMS:%.3f\t' % (
        AMI, ARI, NMI, HOM, completeness, vms, FMS))
    return AMI, ARI, NMI, HOM, completeness, vms, FMS
