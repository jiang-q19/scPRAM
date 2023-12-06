import torch
import torch.nn as nn
from torch import Tensor, optim
from tqdm import tqdm
from scpram.dataset import AnnDataSet
from scpram.utils import balancer
from torch.utils.data import DataLoader
import scanpy as sc
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import ot


class SCPRAM(nn.Module):
    def __init__(self, input_dim=6998, latent_dim=100, hidden_dim=1000,
                 noise_rate=0.1, kl_weight=5e-4, device=None):
        '''
        scPRAM model
        :param input_dim: number of input genes
        :param latent_dim: the number of latent space neurons
        :param hidden_dim: the number of hidden space neurons
        :param noise_rate: the amount of data noise
        :param kl_weight: weight of KL divergence in loss function
        :param device: hardware equipment for training models
        '''
        super(SCPRAM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.noise_rate = noise_rate
        self.kl_weight = kl_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        noise = torch.randn_like(x)
        x_noisy = x + noise * self.noise_rate
        z, mu, logvar = self.encode(x_noisy)
        x_hat = self.decode(z)

        std = torch.exp(logvar / 2)
        loss_kl = kl(
            Normal(mu, std),
            Normal(0, 1)
        ).sum(dim=1)
        loss_rec = ((x - x_hat) ** 2).sum(dim=1)

        return x_hat, loss_rec, loss_kl

    def get_latent_adata(self, adata):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = Tensor(adata.to_df().values).to(device)
        latent_z = self.encode(x)[0].cpu().detach().numpy()
        latent_adata = sc.AnnData(X=latent_z, obs=adata.obs.copy())
        return latent_adata

    def get_loss(self, x):
        x_hat, loss_rec, loss_kl = self.forward(x)
        return loss_rec, loss_kl

    def train_SCPRAM(self, train_adata, epochs=100, batch_size=128, lr=5e-4):
        '''
        train the SCPRAM
        :param train_adata: adata of training set
        :param epochs: the number of training epochs
        :param batch_size: the size of each batch of training data
        :param lr: learning rate
        :return: None
        '''
        device = self.device
        pbar = tqdm(range(epochs))
        anndataset = AnnDataSet(train_adata)
        train_loader = DataLoader(anndataset, batch_size=batch_size, shuffle=True, drop_last=False)
        SCPRAM_loss, loss_rec, loss_kl = 0, 0, 0
        optim_SCPRAM = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        for epoch in pbar:
            pbar.set_description("Training Epoch {}".format(epoch))
            for idx, x in enumerate(train_loader):
                x = x.to(device)
                loss_rec, loss_kl = self.get_loss(x)
                SCPRAM_loss = (0.5 * loss_rec + 0.5 * (loss_kl * self.kl_weight)).mean()
                optim_SCPRAM.zero_grad()
                SCPRAM_loss.backward()
                optim_SCPRAM.step()
            pbar.set_postfix(SCPRAM_loss=SCPRAM_loss.item(), recon_loss=loss_rec.mean().item(),
                             kl_loss=loss_kl.mean().item())

    def ot_predict(self, adata_train, cell_to_pred, key_dic):
        '''
        The optimal transport mapping is used directly to predict the expression state after perturbation
        This method has some effect, but is not accurate enough.
        :param adata_train: adata of training set
        :param cell_to_pred: cell type to predict
        :param key_dic: dictionary of keywords for the data set
        :return: the predicted response after the perturbation
        '''
        ctrl_adata = adata_train[(adata_train.obs[key_dic['condition_key']] == key_dic['ctrl_key'])]
        z_train = self.get_latent_adata(adata_train)
        z_ctrl_adata = z_train[(z_train.obs[key_dic['condition_key']] == key_dic['ctrl_key'])]
        z_stim_adata = z_train[(z_train.obs[key_dic['condition_key']] == key_dic['stim_key'])]
        z_ctrl = z_ctrl_adata.to_df().values
        z_stim = z_stim_adata.to_df().values

        M = ot.dist(z_ctrl, z_stim, metric='euclidean')
        G = ot.emd(torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
                   torch.ones(z_stim.shape[0]) / z_stim.shape[0],
                   torch.tensor(M), numItermax=1000000)
        z_pred = torch.mm(G, torch.tensor(z_stim)).numpy() * z_ctrl.shape[0]

        pred_x = self.decode(Tensor(z_pred).to(self.device)).cpu().detach().numpy()
        pred_adata = sc.AnnData(X=pred_x, obs=ctrl_adata.obs.copy(), var=ctrl_adata.var.copy())
        pred = pred_adata[(pred_adata.obs[key_dic['cell_type_key']]) == cell_to_pred]
        pred.obs[key_dic['condition_key']] = key_dic['pred_key']
        return pred

    def cross_cell_predict(self, train_adata, cell_to_pred, key_dic, n_top=None):
        '''
        Cross-cell type prediction, that is, using pre - and post-perturbation data for the first N known cell types,
        predicts the post-perturbation response of the N+1 cell type. This approach requires the use of cell labels
        and does not require the use of OT to match cells. The prediction worked well.
        :param train_adata: adata of training set
        :param cell_to_pred: cell type to predict
        :param key_dic: dictionary of keywords for the dataset
        :param n_top: what percentage of cells are used for attention mechanisms
        :return: the predicted response after the perturbation
        '''
        ctrl_adata = train_adata[((train_adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                                  (train_adata.obs[key_dic['condition_key']] == key_dic['ctrl_key']))]
        train_z = self.get_latent_adata(train_adata)
        ctrl_z = train_z[(train_z.obs[key_dic['condition_key']] == key_dic['ctrl_key'])]
        stim_z = train_z[(train_z.obs[key_dic['condition_key']] == key_dic['stim_key'])]
        print(ctrl_z.shape, stim_z.shape)
        ctrl_z = balancer(ctrl_z, key_dic['cell_type_key'])
        stim_z = balancer(stim_z, key_dic['cell_type_key'])
        eq = min(ctrl_z.X.shape[0], stim_z.X.shape[0])
        cd_ind = np.random.choice(range(ctrl_z.shape[0]), size=eq, replace=False)
        stim_ind = np.random.choice(range(stim_z.shape[0]), size=eq, replace=False)
        ctrl_z = ctrl_z[cd_ind, :]
        stim_z = stim_z[stim_ind, :]
        types = set(train_adata.obs[key_dic['cell_type_key']])
        ctrl_list, stim_list = [], []
        for cell_type in types:
            if (cell_type == cell_to_pred) or (cell_type == 'isolated'):
                continue
            ctrl_m = ctrl_z[(ctrl_z.obs[key_dic['cell_type_key']] == cell_type)].to_df().values.mean(axis=0)
            stim_m = stim_z[(stim_z.obs[key_dic['cell_type_key']] == cell_type)].to_df().values.mean(axis=0)
            if len(ctrl_list) > 0:
                ctrl_list = np.vstack((ctrl_list, ctrl_m))
                stim_list = np.vstack((stim_list, stim_m))
            else:
                ctrl_list = ctrl_m
                stim_list = stim_m

        delta_list = np.array(stim_list - ctrl_list).reshape(-1, self.latent_dim)
        test_z = self.get_latent_adata(ctrl_adata).to_df().values
        # calulate the cosine similarity
        cos_sim = cosine_similarity(np.array(test_z).reshape(-1, self.latent_dim),
                                    np.array(ctrl_list).reshape(-1, self.latent_dim))
        if n_top is None:
            cos_sim = normalize(cos_sim, axis=1, norm='l1').reshape(test_z.shape[0], -1)
            delta_pred = np.matmul(cos_sim, delta_list)
        else:
            top_indices = np.argsort(cos_sim)[0][-n_top:]
            normalized_weights = cos_sim[0][top_indices] / np.sum(cos_sim[0][top_indices])
            delta_pred = np.sum(normalized_weights[:, np.newaxis] *
                                np.array(delta_list).reshape(-1, self.latent_dim)[top_indices], axis=0)
        pred_z = test_z + delta_pred
        pred_x = self.decode(Tensor(pred_z).to(self.device)).cpu().detach().numpy()
        pred_adata = sc.AnnData(X=pred_x, obs=ctrl_adata.obs.copy(), var=ctrl_adata.var.copy())
        pred_adata.obs[key_dic['condition_key']] = key_dic['pred_key']
        return pred_adata

    def predict(self, train_adata, cell_to_pred, key_dic, ratio=0.05):
        '''
        Method introduced in the paper. OT is used to match the cells before and after perturbation,
        and then the perturbation vector is solved using the attention mechanism.
        :param train_adata: adata of training set including the control data of type to predict
        :param cell_to_pred: cell type to predict
        :param key_dic: dictionary of keywords for the data set
        :param ratio: what percentage of cells are used for attention mechanisms
        :return: the predicted response after the perturbation
        '''
        ctrl_to_pred = train_adata[((train_adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                                    (train_adata.obs[key_dic['condition_key']] == key_dic['ctrl_key']))]
        ctrl_adata = train_adata[(train_adata.obs[key_dic['cell_type_key']] != cell_to_pred) &
                                 (train_adata.obs[key_dic['condition_key']] == key_dic['ctrl_key'])]
        stim_adata = train_adata[(train_adata.obs[key_dic['condition_key']] == key_dic['stim_key'])]

        ctrl = self.get_latent_adata(ctrl_adata).to_df().values
        stim = self.get_latent_adata(stim_adata).to_df().values
        M = ot.dist(stim, ctrl, metric='euclidean')
        G = ot.emd(torch.ones(stim.shape[0]) / stim.shape[0],
                   torch.ones(ctrl.shape[0]) / ctrl.shape[0],
                   torch.tensor(M), numItermax=100000)
        match_idx = torch.max(G, 0)[1].numpy()
        stim_new = stim[match_idx]
        delta_list = stim_new - ctrl
        test_z = self.get_latent_adata(ctrl_to_pred).to_df().values
        cos_sim = cosine_similarity(np.array(test_z).reshape(-1, self.latent_dim),
                                    np.array(ctrl).reshape(-1, self.latent_dim))
        n_top = int(np.ceil(ctrl.shape[0] * ratio))
        top_indices = np.argsort(cos_sim)[0][-n_top:]
        normalized_weights = cos_sim[0][top_indices] / np.sum(cos_sim[0][top_indices])
        delta_pred = np.sum(normalized_weights[:, np.newaxis] *
                            np.array(delta_list).reshape(-1, self.latent_dim)[top_indices], axis=0)
        pred_z = test_z + delta_pred
        pred_x = self.decode(Tensor(pred_z).to(self.device)).cpu().detach().numpy()
        pred_adata = sc.AnnData(X=pred_x, obs=ctrl_to_pred.obs.copy(), var=ctrl_to_pred.var.copy())
        pred_adata.obs[key_dic['condition_key']] = key_dic['pred_key']
        return pred_adata
