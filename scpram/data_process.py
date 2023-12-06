import scanpy as sc
import numpy as np


def adata_process(adata, min_genes=200, min_cells=10, max_value=10, n_top_genes=6000):
    '''
    Process the raw count matrix of the input
    :param adata: Primitive counting matrix
    :param min_genes: Each cell has at least m genes expressed, otherwise they are filtered
    :param min_cells: Each gene is expressed in at least n cells, otherwise it is filtered
    :param max_value: The maximum value to which the gene expression value is normalized
    :param n_top_genes: The maximum number of differential genes retained
    :return:
    '''
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=max_value)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]
    df = adata.to_df().clip(lower=0)
    adata.X = df
    return adata


# 按照一定的比例将基因表达矩阵的部分非零值置为0
def add_mask(adata, fraction):
    '''
    Setting the non-zero value of the gene expression matrix to 0 in a certain proportion
    :param adata: adata of input
    :param fraction: the proportion that is set to 0
    :return: adata processed
    '''
    df = adata.to_df()
    idx = df.index
    for i in range(len(df)):
        non_zero_indices = df.iloc[i].index[df.iloc[i].values != 0]
        num_to_zero = int(len(non_zero_indices) * fraction)
        indices_to_zero = np.random.choice(non_zero_indices, size=num_to_zero, replace=False)
        df.loc[idx[i], indices_to_zero] = 0
    adata.X = df
    return adata