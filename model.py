import sys
import os
import time
import pickle
import argparse
import ast
import multiprocessing
import importlib
from multiprocessing.pool import Pool

import numpy as np
import numpy.linalg as LA

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from tqdm.auto import tqdm
from torchvision.transforms import Normalize

from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    umap = None
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
#                   Configuration Container
# ---------------------------------------------------------------------------
class ConfigClass:
    pass

Config = None  # Will be set inside main().


# ---------------------------------------------------------------------------
#                          Dataset Class
# ---------------------------------------------------------------------------
class Subtomogram_Dataset:
    """
    Basic PyTorch dataset that returns (sample, label_one_hot).
    """
    def __init__(self, train_data, label_one_hot):
        self.train_data = train_data
        self.label_one_hot = label_one_hot

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        features = self.train_data[index]
        labels = self.label_one_hot[index]
        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(labels)
        return features, labels


# ---------------------------------------------------------------------------
#                           Utility Functions
# ---------------------------------------------------------------------------
def pickle_dump(obj, path, protocol=2):
    """Convenience wrapper for pickle dumping."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

def load_pickle_file(path):
    """Convenience wrapper for pickle loading."""
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def run_iterator(tasks, worker_num=multiprocessing.cpu_count(), verbose=False):
    """
    Possibly parallel execution of tasks in a dict; originally from DISCA code.
    """
    if verbose:
        print('parallel_multiprocessing()', 'start', time.time())

    worker_num = min(worker_num, multiprocessing.cpu_count())
    for i,t in tasks.items():
        if 'args' not in t:
            t['args'] = ()
        if 'kwargs' not in t:
            t['kwargs'] = {}
        if 'id' not in t:
            t['id'] = i
        assert t['id'] == i

    completed_count = 0
    if worker_num > 1:
        pool = Pool(processes=worker_num)
        pool_apply = []
        for i,t in tasks.items():
            aa = pool.apply_async(func=call_func, kwds={'t':t})
            pool_apply.append(aa)

        for pa in pool_apply:
            yield pa.get(99999)
            completed_count += 1
            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()
        pool.close()
        pool.join()
        del pool
    else:
        for i,t in tasks.items():
            yield call_func(t)
            completed_count += 1
            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()

    if verbose:
        print('parallel_multiprocessing()', 'end', time.time())

# alias
run_batch = run_iterator

def call_func(t):
    """Helper for run_iterator."""
    if 'func' in t:
        assert 'module' not in t
        assert 'method' not in t
        func = t['func']
    else:
        modu = importlib.import_module(t['module'])
        func = getattr(modu, t['method'])
    r = func(*t['args'], **t['kwargs'])
    return {'id': t['id'], 'result': r}


# ---------------------------------------------------------------------------
#            NEW: GMM PDF & LOG-LIKELIHOOD USING PRECOMPUTED INVERSES
# ---------------------------------------------------------------------------

def multivariate_gaussian_pdf_with_inv(x, mu, Sigma_inv, det_Sigma):
    """
    Evaluate multivariate normal density at x for distribution N(mu, Sigma)
    given that Sigma_inv = inv(Sigma), and det_Sigma = det(Sigma) were
    precomputed (to avoid repeated inversion).
    """
    d = len(mu)
    diff = x - mu
    # Mahalanobis distance
    expo = diff @ Sigma_inv @ diff
    # normalizing constant
    norm_const = 1.0 / np.sqrt((2.0 * np.pi)**d * det_Sigma + 1e-35)
    return norm_const * np.exp(-0.5 * expo)


def compute_log_likelihood_with_inv(X, pis, mus, Sigmas_inv, Sigmas_det):
    """
    Compute log-likelihood of data X under the GMM params,
    using the precomputed inverses & determinants for each Sigma_k.
    """
    N, d = X.shape
    K = pis.shape[0]
    log_likelihood = 0.0

    for n in range(N):
        p_sum = 0.0
        x_n = X[n]
        for k in range(K):
            p_sum += pis[k] * multivariate_gaussian_pdf_with_inv(
                x_n, mus[k], Sigmas_inv[k], Sigmas_det[k]
            )
        log_likelihood += np.log(p_sum + 1e-35)
    return log_likelihood


# ---------------------------------------------------------------------------
#        Modified GMM & Semi-Supervised Clustering (One-Time Inversion)
# ---------------------------------------------------------------------------

def custom_semi_supervised_gmm_fit(
    X,
    labeled_indices,
    labeled_labels,
    K_total,
    reg_covar=1e-5,
    max_iter=100,
    tol=1e-3,
    verbose=True
):
    """
    Custom semi-supervised GMM:
      - Labeled data are locked to known classes.
      - For total K=K_total, we have K_known classes from the labeled data,
        plus K_new = K_total - K_known "new" clusters to discover among unlabeled data.

    Return:
      responsibilities (N,K_total), pis, mus, Sigmas, final_ll
    """
    X = np.array(X)  # ensure numpy
    N, d = X.shape
    labeled_indices = np.array(labeled_indices)
    labeled_labels = np.array(labeled_labels)
    unique_classes = np.unique(labeled_labels)
    K_known = len(unique_classes)
    print(f"K_known:{K_known}")

    if K_total < K_known:
        raise ValueError(f"K_total={K_total} < number of known classes={K_known}!")

    # separate unlabeled
    all_inds = np.arange(N)
    unlabeled_indices = np.setdiff1d(all_inds, labeled_indices)

    # 1) Initialize GMM params
    pis = np.zeros(K_total)
    mus = np.zeros((K_total, d))
    Sigmas = np.zeros((K_total, d, d))

    # For known classes:
    for local_idx, c in enumerate(unique_classes):
        c_points = X[labeled_indices[labeled_labels == c]]
        if c_points.shape[0] < 1:
            # degenerate case: no points => random
            mus[local_idx] = np.random.randn(d)
            Sigmas[local_idx] = np.eye(d)
            pis[local_idx] = 1e-6
        else:
            mus[local_idx] = np.mean(c_points, axis=0)
            cov = np.cov(c_points.T) + reg_covar*np.eye(d)
            Sigmas[local_idx] = cov
            # weight is proportion among entire dataset:
            pis[local_idx] = float(c_points.shape[0]) / float(N)

    # For new unknown clusters:
    K_new = K_total - K_known
    for j in range(K_new):
        cluster_id = K_known + j
        r_pt = X[np.random.randint(0, N)]
        mus[cluster_id] = r_pt
        Sigmas[cluster_id] = np.cov(X.T) + reg_covar*np.eye(d)
        pis[cluster_id] = 1e-6
    # normalize mixing weights
    pis_sum = pis.sum()
    if pis_sum < 1e-12:
        pis = np.ones(K_total)/K_total
    else:
        pis /= pis_sum

    # 2) EM with locked labeled assignments
    responsibilities = np.zeros((N, K_total))
    prev_ll = None
    for iter_ in range(max_iter):

        # ----------------------------
        # E-step
        # ----------------------------
        # Lock labeled data
        responsibilities[labeled_indices] = 0.0

        unique_classes = np.unique(labeled_labels)
        label_to_cluster = {label: idx for idx, label in enumerate(unique_classes)}

        for i, c in zip(labeled_indices, labeled_labels):

            cluster_id = label_to_cluster[c]
            responsibilities[i, cluster_id] = 1.0

            #responsibilities[i, c] = 1.0

        # Precompute the inverses & determinants once *before* computing PDFs
        # for the unlabeled data. This is the key optimization.
        Sigmas_inv = np.zeros_like(Sigmas)
        Sigmas_det = np.zeros(K_total)
        for k in range(K_total):
            try:
                inv_k = LA.inv(Sigmas[k])
                det_k = LA.det(Sigmas[k])
            except LA.LinAlgError:
                # fallback if singular
                inv_k = np.eye(d)
                det_k = 1.0
            Sigmas_inv[k] = inv_k
            Sigmas_det[k] = det_k

        # For unlabeled data
        for i in unlabeled_indices:
            x_i = X[i]
            unnorm = np.zeros(K_total)
            for k in range(K_total):
                unnorm[k] = pis[k] * multivariate_gaussian_pdf_with_inv(
                    x_i, mus[k], Sigmas_inv[k], Sigmas_det[k]
                )
            denom = unnorm.sum() + 1e-35
            responsibilities[i, :] = unnorm / denom

        # ----------------------------
        # M-step
        # ----------------------------
        Nk = responsibilities.sum(axis=0)
        pis = Nk / float(N)

        # update mu
        for k in range(K_total):
            if Nk[k] < 1e-12:
                mus[k] = np.random.randn(d)
                Sigmas[k] = np.eye(d)
                pis[k] = 1e-6
                continue
            sum_ = np.zeros(d)
            for i in range(N):
                sum_ += responsibilities[i,k] * X[i]
            mus[k] = sum_ / Nk[k]

        # update Sigma
        for k in range(K_total):
            if Nk[k] < 1e-12:
                continue
            diff_sum = np.zeros((d,d))
            for i in range(N):
                diff = (X[i] - mus[k]).reshape(-1,1)
                diff_sum += responsibilities[i,k] * (diff @ diff.T)
            cov_k = diff_sum / Nk[k]
            cov_k += reg_covar*np.eye(d)
            Sigmas[k] = cov_k

        # Now compute log-likelihood with the updated Sigmas.
        # We'll re-compute Sigmas_inv/det for the log-likelihood call.
        Sigmas_inv_ll = np.zeros_like(Sigmas)
        Sigmas_det_ll = np.zeros(K_total)
        for k in range(K_total):
            try:
                Sigmas_inv_ll[k] = LA.inv(Sigmas[k])
                Sigmas_det_ll[k] = LA.det(Sigmas[k])
            except LA.LinAlgError:
                Sigmas_inv_ll[k] = np.eye(d)
                Sigmas_det_ll[k] = 1.0

        ll_ = compute_log_likelihood_with_inv(X, pis, mus, Sigmas_inv_ll, Sigmas_det_ll)

        if verbose:
            print(f"Iteration={iter_}, LogLik={ll_:.3f}")

        if prev_ll is not None and abs(ll_ - prev_ll) < tol:
            break
        prev_ll = ll_

    return responsibilities, pis, mus, Sigmas, prev_ll


def custom_semi_supervised_gmm_multiK(
    X,
    labeled_indices,
    labeled_labels,
    minK,
    maxK,
    reg_covar=1e-5,
    max_iter=100,
    tol=1e-3,
    verbose=True
):
    """
    Try each K in [minK, maxK]. For each K, run custom_semi_supervised_gmm_fit.
    Compute BIC, pick best K. Return (best_resp, best_pis, best_mus, best_Sigmas, best_ll, best_K).
    """
    N, d = X.shape
    K_known = len(np.unique(labeled_labels))
    best_bic = float('inf')
    best_params = None
    best_K = None
    print(f"K_known:{K_known}")

    for K in range(minK, maxK+1):
        if K < K_known:
            # skip impossible scenario
            continue

        resp, pis, mus, Sigmas, ll_ = custom_semi_supervised_gmm_fit(
            X=X,
            labeled_indices=labeled_indices,
            labeled_labels=labeled_labels,
            K_total=K,
            reg_covar=reg_covar,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )
        # free params: (K - 1) + K*d + K*d(d+1)//2
        free_params = (K - 1) + K*d + K*(d*(d+1)//2)

        # We must compute the final log-likelihood again with inverses, or we can
        # rely on ll_ from the last iteration. We'll do a final BIC using that ll_:
        bic_value = -2.0* ll_ + free_params*np.log(N)

        if verbose:
            print(f"K={K}, LogLik={ll_:.3f}, BIC={bic_value:.3f}")

        if bic_value < best_bic:
            best_bic = bic_value
            best_params = (resp, pis, mus, Sigmas, ll_)
            best_K = K

    if best_params is None:
        raise ValueError("No valid K found in the given range!")
    if verbose:
        print(f"\nBest K={best_K}, BIC={best_bic:.3f}")

    resp, pis, mus, Sigmas, ll_ = best_params
    return resp, pis, mus, Sigmas, ll_, best_K


# ---------------------------------------------------------------------------
#               Additional DISCA Utility Functions
# ---------------------------------------------------------------------------
def remove_empty_cluster(labels):
    """
    Re-label cluster IDs to be contiguous [0..C-1].
    """
    labels_unique = np.unique(labels)
    for i in range(len(labels_unique)):
        labels[labels == labels_unique[i]] = i
    return labels

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

def data_augmentation(x_train, factor=2):
    """
    Example 3D data augmentation from original DISCA code:
    random 3D rotation, etc.
    """
    if factor > 1:
        x_train_augmented = []
        x_train_augmented.append(x_train)
        for f in range(1, factor):
            ts = {}
            for i in range(len(x_train)):
                t = {}
                t['func'] = rotate3d_zyz
                args_t = {}
                args_t['data'] = x_train[i,:,:,:,0]
                args_t['Inv_R'] = random_rotation_matrix()
                t['kwargs'] = args_t
                ts[i] = t
            rs = run_batch(ts, worker_num=4)  # adjust as needed
            x_train_f = np.expand_dims(np.array([_['result'] for _ in rs]), -1)
            x_train_augmented.append(x_train_f)
        x_train_augmented = np.concatenate(x_train_augmented)
    else:
        x_train_augmented = x_train
        # This line replaces zeros with random normal noise
        x_train[x_train == 0] = np.random.normal(loc=0.0, scale=1.0, size=np.sum(x_train == 0))
    return x_train_augmented

def random_rotation_matrix():
    m = np.random.random((3,3))
    u, s, v = LA.svd(m)
    return u

def rotate3d_zyz(data, Inv_R, center=None, order=2):
    from scipy import mgrid
    from scipy.ndimage import map_coordinates

    if center is None:
        cx = data.shape[0] / 2
        cy = data.shape[1] / 2
        cz = data.shape[2] / 2
    else:
        (cx, cy, cz) = center

    grid = mgrid[-cx:data.shape[0]-cx, -cy:data.shape[1]-cy, -cz:data.shape[2]-cz]
    temp = grid.reshape((3, -1))
    temp = np.dot(Inv_R, temp)
    grid = np.reshape(temp, grid.shape)
    grid[0] += cx
    grid[1] += cy
    grid[2] += cz
    d = map_coordinates(data, grid, order=order)
    return d

def prepare_training_data(x_train, labels, label_smoothing_factor):
    """
    Prepares data for YOPO training by data augmentation and label smoothing.
    """
    label_one_hot = one_hot(labels, len(np.unique(labels)))
    index = np.array(range(x_train.shape[0] * Config.factor_use))
    np.random.shuffle(index)
    x_train_augmented = data_augmentation(x_train, Config.factor_use)
    x_train_permute = x_train_augmented[index].copy()

    label_smoothing_factor *= 0.9
    labels_augmented = np.tile(smooth_labels(label_one_hot, label_smoothing_factor),
                               (Config.factor_use,1))
    labels_permute = labels_augmented[index].copy()
    return label_one_hot, x_train_permute, label_smoothing_factor, labels_permute


# Davies-Bouldin index for cluster validation (DISCA approach).
def DDBI(features, labels):
    means_init = np.array([np.mean(features[labels == i], 0) for i in np.unique(labels)])
    precisions_init = []
    for i in np.unique(labels):
        cov_ = np.cov(features[labels == i].T) + Config.reg_covar * np.eye(features.shape[1])
        prec_ = LA.inv(cov_)
        precisions_init.append(prec_)
    precisions_init = np.array(precisions_init)

    T = []
    for i, c_ in enumerate(np.unique(labels)):
        cluster_feats = features[labels == c_]
        mu_ = means_init[i]
        prec_ = precisions_init[i]
        diffs = (cluster_feats - mu_)
        val_ = np.mean(np.diag(diffs @ prec_ @ diffs.T))
        T.append(val_)
    T = np.array(T)

    # pairwise distances
    D = []
    for i, c_i in enumerate(np.unique(labels)):
        row = []
        for j, c_j in enumerate(np.unique(labels)):
            diff = means_init[j] - means_init[i]
            val_ij = diff @ precisions_init[i] @ diff.T
            row.append(val_ij)
        D.append(row)
    D = np.array(D)

    # compute DBI
    K_ = len(np.unique(labels))
    DBI_matrix = np.zeros((K_, K_))
    for i in range(K_):
        for j in range(K_):
            if i != j:
                DBI_matrix[i,j] = (T[i] + T[j]) / (D[i,j] + D[j,i])
    DBI = np.mean(np.max(DBI_matrix, axis=0))
    return DBI


# ---------------------------------------------------------------------------
#                       YOPO Model Definitions
# ---------------------------------------------------------------------------
class YOPOFeatureModel(nn.Module):
    def __init__(self):
        super(YOPOFeatureModel, self).__init__()

        self.dropout = nn.Dropout(0.5)
        self.m1 = self.get_block(1, 64)
        self.m2 = self.get_block(64, 80)
        self.m3 = self.get_block(80, 96)
        self.m4 = self.get_block(96, 112)
        self.m5 = self.get_block(112, 128)
        self.m6 = self.get_block(128, 144)
        self.m7 = self.get_block(144, 160)
        self.m8 = self.get_block(160, 176)
        self.m9 = self.get_block(176, 192)
        self.m10 = self.get_block(192, 208)
        self.batchnorm = nn.BatchNorm3d(1360)
        self.linear = nn.Linear(1360, 1024)
        self.weight_init(self)

    def forward(self, input_image):
        output = input_image.view(-1, 1, Config.image_size, Config.image_size, Config.image_size)
        output = self.dropout(output)
        output = self.m1(output)
        o1 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m2(output)
        o2 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m3(output)
        o3 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m4(output)
        o4 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m5(output)
        o5 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m6(output)
        o6 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m7(output)
        o7 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m8(output)
        o8 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m9(output)
        o9 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m10(output)
        o10 = F.max_pool3d(output, kernel_size=output.size()[2:])

        m = torch.cat((o1, o2, o3, o4, o5, o6, o7, o8, o9, o10), dim=1)
        m = self.batchnorm(m)
        m = nn.Flatten()(m)
        m = self.linear(m)
        return m

    @staticmethod
    def get_block(input_channels, output_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      padding=0),
            nn.ELU(),
            nn.BatchNorm3d(output_channels)
        )

    @staticmethod
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)


class YOPOClassification(nn.Module):
    def __init__(self, num_labels, vector_size=1024):
        super(YOPOClassification, self).__init__()
        self.main_input = nn.Linear(vector_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.main_input(x)
        x = self.softmax(x)
        return x


class YOPO_Final_Model(nn.Module):
    def __init__(self, yopo_feature, yopo_classification):
        super(YOPO_Final_Model, self).__init__()
        self.feature_model = yopo_feature
        self.classification_model = yopo_classification

    def forward(self, input_image):
        features = self.feature_model(input_image)
        output = self.classification_model(features)
        return output


# ---------------------------------------------------------------------------
#                   YOPO Output Layer Updating
# ---------------------------------------------------------------------------
def update_output_layer(K, label_one_hot, batch_size, model_feature, features, lr, verbose=True):
    """
    If K changed, or first time, re-initialize a YOPOClassification with K classes
    and train it briefly on the (features, label_one_hot) pairs.
    """
    print('Updating output layer to K=', K)
    model_classification = YOPOClassification(num_labels=K).to(Config.device)

    optim_ = torch.optim.NAdam(model_classification.parameters(), lr=0.0001,
                               betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.MultiMarginLoss()

    dataset = Subtomogram_Dataset(features, label_one_hot)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(Config.output_layer_iteration):
        model_classification.train()
        train_total = 0
        train_correct = 0
        epoch_loss = 0.0
        start_time = time.time()

        pbar = tqdm(loader, desc=f'UpdateOutput Epoch {epoch+1}/{Config.output_layer_iteration}')
        for feat_, lab_ in pbar:
            feat_ = feat_.to(Config.device)
            lab_ = lab_.to(Config.device)

            pred_ = model_classification(feat_)
            optim_.zero_grad()

            # multi-margin requires integer labels
            lab_int = torch.argmax(lab_, 1)
            loss_ = criterion(pred_, lab_int)
            loss_.backward()
            optim_.step()

            epoch_loss += loss_.item()
            preds_i = torch.argmax(pred_, 1)
            train_correct += (preds_i == lab_int).sum().item()
            train_total += lab_int.size(0)

        exec_time = time.time() - start_time
        accuracy = train_correct / train_total
        if verbose:
            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={accuracy:.4f}, Time={exec_time:.2f}s")

    # Combine new classification with existing feature model
    model = YOPO_Final_Model(model_feature, model_classification)
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr,
                                  betas=(0.9,0.999), eps=1e-08)
    criterion = nn.MultiMarginLoss()

    return model, optimizer, criterion


# ---------------------------------------------------------------------------
#                         Main DISCA-Style Script
# ---------------------------------------------------------------------------

# ===================== New Additions =====================
from scipy.stats import mode
from sklearn.metrics.pairwise import cosine_similarity

def compute_entropy(responsibilities):
    entropy = -np.sum(responsibilities * np.log(responsibilities + 1e-12), axis=1)
    norm_entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
    confidence = 1 - norm_entropy
    return confidence

def propagate_labels(features, label_one_hot, confidence, k=5, threshold=0.8):
    sim = cosine_similarity(features)
    confident_idx = confidence > threshold
    label_prop = np.copy(label_one_hot)
    for i in range(len(features)):
        if confidence[i] < threshold:
            top_k = sim[i, confident_idx].argsort()[-k:]
            neighbor_idx = np.where(confident_idx)[0][top_k]
            label_prop[i] = label_one_hot[neighbor_idx].mean(axis=0)
    return label_prop

def pca_voting_clustering(features, labeled_indices, labeled_labels, minK, maxK):
    pca_dims = [8, 16, 32]
    label_votes = []
    for d in pca_dims:
        pca_proj = PCA(n_components=d).fit_transform(features)
        resp, _, _, _, _, _ = custom_semi_supervised_gmm_multiK(
            X=pca_proj,
            labeled_indices=labeled_indices,
            labeled_labels=labeled_labels,
            minK=minK,
            maxK=maxK,
            reg_covar=Config.reg_covar,
            max_iter=30,
            tol=1e-3,
            verbose=False
        )
        label_votes.append(np.argmax(resp, axis=1))
    labels_stack = np.stack(label_votes, axis=1)
    labels_voted = mode(labels_stack, axis=1).mode.flatten()
    return labels_voted
# ========================================================


def build_cluster_cmap(num_clusters):
    """Build a color map large enough for the current number of clusters."""
    base = list(plt.cm.get_cmap("tab20").colors)
    if num_clusters <= len(base):
        color_list = base[:num_clusters]
    else:
        extra_needed = num_clusters - len(base)
        hsv_extra = plt.cm.hsv(np.linspace(0.05, 0.95, extra_needed))
        color_list = base + list(hsv_extra)
    return plt.cm.colors.ListedColormap(color_list[:num_clusters])


def save_embedding_plot(embedding, labels, method_name, iteration_idx, checkpoint_dir):
    """Save a 2D embedding scatter plot colored by the current iteration labels."""
    labels = np.asarray(labels)
    num_clusters = len(np.unique(labels))
    cmap = build_cluster_cmap(num_clusters)

    out_dir = os.path.join(checkpoint_dir, f"{method_name.lower()}_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method_name.lower()}_iter_{iteration_idx}.png")

    plt.figure(figsize=(5, 5), dpi=150)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=4, cmap=cmap)
    plt.title(f"{method_name} | Iter {iteration_idx} | K={num_clusters}")
    plt.axis('on')
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"{method_name} saved -> {out_path}")


def generate_embedding_plots(features_array, labels, iteration_idx, checkpoint_dir, max_points=5000):
    """Generate PCA, t-SNE, and optionally UMAP plots for the current iteration."""
    n_vis = min(max_points, features_array.shape[0])
    idx_vis = np.random.choice(features_array.shape[0], n_vis, replace=False)
    feat_vis = features_array[idx_vis]
    labels_vis = np.asarray(labels)[idx_vis]

    pca_2d = PCA(n_components=2).fit_transform(feat_vis)
    save_embedding_plot(pca_2d, labels_vis, "PCA", iteration_idx, checkpoint_dir)

    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    tsne_2d = tsne.fit_transform(feat_vis)
    save_embedding_plot(tsne_2d, labels_vis, "TSNE", iteration_idx, checkpoint_dir)

    if umap is None:
        print("UMAP requested but umap-learn is not installed; skipping UMAP plot.")
    else:
        reducer = umap.UMAP(n_components=2, init="spectral", random_state=42)
        umap_2d = reducer.fit_transform(feat_vis)
        save_embedding_plot(umap_2d, labels_vis, "UMAP", iteration_idx, checkpoint_dir)


def main():
    parser = argparse.ArgumentParser(description="DISCA Semi-Supervised with Multi-K & Fixed Labeled Subset")

    # --- Key I/O & Configuration ---
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Path to save final YOPO model.")
    parser.add_argument("--output_label_path", type=str, required=True,
                        help="Path to save final cluster labels.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory to store intermediate checkpoints.")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to load from an existing checkpoint.")
    parser.add_argument("--outlogfile", type=str, required=True,
                        help="Log file to append progress info.")

    # --- Data & Known Labels ---
    parser.add_argument(
        "--training_data_path",
        type=str,
        required=True,
        help="Pickle with shape (N, D1, D2, D3) or (N, D1, D2, D3, 1)."
    )

    parser.add_argument(
        "--labeled_indices_path",
        type=str,
        required=True,
        help="Pickle with a 1D array of labeled sample indices."
    )

    parser.add_argument(
        "--labeled_labels_path",
        type=str,
        required=True,
        help="Pickle with known cluster ID for each labeled sample index."
    )

    # --- Program Flow ---
    parser.add_argument("--M", type=int, default=100,
                        help="Max number of outer DISCA iterations.")
    parser.add_argument("--yopo_iteration", type=int, default=2,
                        help="Number of epochs to train YOPO each iteration.")
    parser.add_argument("--output_layer_iteration", type=int, default=10,
                        help="Epochs to train classification head after K changes.")

    # --- Model / Data Params ---
    parser.add_argument("--img_size", type=int, default=32,
                        help="Size of each dimension of the 3D subtomogram.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--factor_use", type=int, default=2,
                        help="Data augmentation factor.")
    parser.add_argument("--normalize", type=bool, default=False,
                        help="If True, apply normalization to the input data.")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.2)
    parser.add_argument("--reg_covar", type=float, default=1e-5,
                        help="Regularization in GMM covariance.")
    parser.add_argument("--candidateKs", type=str, required=True,
                        help="Comma-separated minK,maxK. E.g. '2,10' => [2..10].")
    parser.add_argument("--enable_plots", action="store_true",
                        help="If set, save PCA, t-SNE, and UMAP plots for selected iterations.")
    parser.add_argument("--plot_max_points", type=int, default=5000,
                        help="Maximum number of points to subsample for each embedding plot.")

    # --- GT for evaluation (optional) ---
    parser.add_argument("--gt_known", type=bool, default=True,
                        help="If True, we have external ground-truth for entire dataset.")
    parser.add_argument("--path_to_gt", type=str, default=None,
                        help="Optional pickle with ground-truth for all samples, used for homogeneity, v-measure, etc.")
    parser.add_argument(
        "--class_id",
        type=ast.literal_eval,
        default=None,
        help="Optional dictionary {class_name: int_label} if you need to map GT strings to integers."
    )
    parser.add_argument("--true_k", type=int, default=None,
                        help="Number of classes in ground truth (optional).")

    args = parser.parse_args()

    # Make checkpoint_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Prepare Config
    global Config
    Config = ConfigClass()
    Config.image_size = args.img_size
    # parse candidateKs => minK, maxK
    parts = args.candidateKs.split(",")
    minK = int(parts[0])
    maxK = int(parts[1])

    Config.batch_size = args.batch_size
    Config.M = args.M
    Config.lr = args.lr
    Config.label_smoothing_factor = args.label_smoothing_factor
    Config.reg_covar = args.reg_covar
    Config.model_path = args.output_model_path
    Config.label_path = args.output_label_path
    Config.factor_use = args.factor_use
    Config.yopo_iteration = args.yopo_iteration
    Config.output_layer_iteration = args.output_layer_iteration
    Config.enable_plots = args.enable_plots
    Config.plot_max_points = args.plot_max_points
    Config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    x_train_raw = load_pickle_file(args.training_data_path)
    if args.normalize:
        def image_normalization(img_list):
            normalized_images = []
            for image in img_list:
                arr = np.array(image)
                arr_t = torch.tensor(arr)
                norm_ = Normalize(mean=[arr_t.mean()], std=[arr_t.std()])(arr_t).tolist()
                normalized_images.append(norm_)
            return normalized_images
        x_train_raw = image_normalization(x_train_raw)

    x_train_raw = np.array(x_train_raw)
    # Ensure shape => (N, D1, D2, D3, 1)
    if len(x_train_raw.shape) == 4:
        x_train_raw = np.expand_dims(x_train_raw, -1)

    print("x_train_raw.shape:", x_train_raw.shape)
    N = x_train_raw.shape[0]

    x_train = torch.tensor(x_train_raw, dtype=torch.float32)
    # Possibly load ground-truth for evaluation
    gt = None
    if args.gt_known and args.path_to_gt is not None:
        raw_gt = load_pickle_file(args.path_to_gt)
        if args.class_id is not None:
            #numerical_gt = [args.class_id.get(lbl, -1) for lbl in raw_gt]
            #gt = np.array(numerical_gt)
            gt = np.array(raw_gt)
        else:
            gt = np.array(raw_gt)

    # -----------------------------------------------------
    # Labeled Indices & Labels
    # -----------------------------------------------------
    labeled_indices = None
    labeled_labels = None
    if args.labeled_indices_path and args.labeled_labels_path:
        labeled_indices = load_pickle_file(args.labeled_indices_path)
        labeled_labels = load_pickle_file(args.labeled_labels_path)
        if len(labeled_indices) != len(labeled_labels):
            raise ValueError("Mismatch in number of labeled_indices vs. labeled_labels!")
        print(f"Labeled subset size = {len(labeled_indices)}")

    # -----------------------------------------------------
    # Main Variables
    # -----------------------------------------------------
    i = 0
    done = False
    DBI_best = float('inf')
    labels = None
    model = None

    # If loading from checkpoint
    if args.load_checkpoint is not None and os.path.exists(args.load_checkpoint):
        checkpoint = torch.load(args.load_checkpoint)
        model = checkpoint['model']
        labels = checkpoint['labels']
        DBI_best = checkpoint['DBI_best']
        i = checkpoint['iteration']
        done = checkpoint['done']
        print(f"Loaded checkpoint from iteration={i}")
    else:
        print("Starting training from scratch")

    total_loss = []
    with open(args.outlogfile, 'a', buffering=1) as logfile:
        while not done and i < Config.M:
            print(f"\n========== Iteration: {i} ==========")

            # 1) Feature Extraction
            if i == 0 or model is None:
                print("Creating new YOPOFeatureModel...")
                model_feature = YOPOFeatureModel().to(Config.device)
            else:
                print("Extracting feature_model from existing model...")
                model_feature = model.feature_model

            model_feature.eval()
            features_array = []
            loader_feat = DataLoader(x_train, batch_size=Config.batch_size, shuffle=False)
            with torch.no_grad():
                for batch_ in tqdm(loader_feat, desc="Extracting Features"):
                    batch_ = batch_.to(Config.device)
                    feats_ = model_feature(batch_).cpu().numpy()
                    features_array.append(feats_)
            features_array = np.concatenate(features_array, axis=0)  # shape (N,1024)

            # 2) Cluster initialisation / refinement
            if labeled_indices is not None and labeled_labels is not None:
                pca = PCA(n_components=16)
                features_16 = pca.fit_transform(features_array)

                if i == 0:
                    print("Boot-strapping clusters with PCA voting.")
                    labels_temp = pca_voting_clustering(
                        features_array,
                        labeled_indices,
                        labeled_labels,
                        minK,
                        maxK
                    )
                    K = len(np.unique(labels_temp))
                else:
                    resp, pis, mus, Sigmas, final_ll, best_K = custom_semi_supervised_gmm_multiK(
                        X               = features_16,
                        labeled_indices = labeled_indices,
                        labeled_labels  = labeled_labels,
                        minK            = minK,
                        maxK            = maxK,
                        reg_covar       = Config.reg_covar,
                        max_iter        = 30,   # fewer iterations for speed
                        tol             = 1e-3,
                        verbose         = False,
                    )

                    # Confidence-based label refinement
                    confidence    = compute_entropy(resp)
                    label_one_hot = one_hot(np.argmax(resp, 1), best_K)
                    label_one_hot = propagate_labels(
                        features_16,
                        label_one_hot,
                        confidence,
                        k         = 5,
                        threshold = 0.8
                    )
                    labels_temp = np.argmax(label_one_hot, 1)
                    K = best_K
            else:
                raise ValueError("No labeled data provided. Please provide labeled subset to fix known labels.")

            labels_temp = remove_empty_cluster(labels_temp)

            if Config.enable_plots and i > 0:
                generate_embedding_plots(
                    features_array=features_array,
                    labels=labels_temp,
                    iteration_idx=i,
                    checkpoint_dir=args.checkpoint_dir,
                    max_points=Config.plot_max_points
                )

            # 3) DBI
            K_est = len(np.unique(labels_temp))
            print(f"Chosen K={K_est}, cluster sizes:", [np.sum(labels_temp == c_) for c_ in np.unique(labels_temp)])
            pca_ = PCA(n_components=16).fit_transform(features_array)
            DBI_value = DDBI(pca_, labels_temp)
            print(f"Distortion-based DBI={DBI_value:.4f}")

            # 4) If DBI is best, save
            if DBI_value < DBI_best:
                print("New best DBI; saving model/labels.")
                DBI_best = DBI_value
                pickle_dump(labels_temp, Config.label_path)

            labels = labels_temp

            # 5) Prepare YOPO training data
            print("Preparing YOPO training data for next iteration...")
            label_one_hot, x_train_permute, label_smoothing_factor, labels_permute = prepare_training_data(
                x_train_raw, labels, Config.label_smoothing_factor
            )

            # 6) Update classification layer
            model, optim_, crit_ = update_output_layer(
                K=K_est,
                label_one_hot=label_one_hot,
                batch_size=Config.batch_size,
                model_feature=model_feature,
                features=features_array,  # you might do PCA again here if you want
                lr=Config.lr,
                verbose=True
            )

            # 7) CNN Training
            print("Fine-tuning YOPO (feature+classification) with new cluster assignments.")
            sched_ = StepLR(optim_, step_size=1, gamma=0.95)

            ds_train = Subtomogram_Dataset(x_train_permute, labels_permute)
            dl_train = DataLoader(ds_train, batch_size=Config.batch_size, shuffle=True)

            model.train()
            iteration_loss = 0.0
            train_correct = 0
            train_total = 0
            start_time = time.time()

            for ep in range(Config.yopo_iteration):
                sched_.step()
                print(f"YOPO epoch {ep+1}/{Config.yopo_iteration}, LR={sched_.get_last_lr()}")

                for Xb_, yb_ in tqdm(dl_train, desc="Training CNN"):
                    Xb_ = Xb_.to(Config.device)
                    yb_ = yb_.to(Config.device)
                    preds_ = model(Xb_)
                    optim_.zero_grad()

                    yb_int = torch.argmax(yb_, 1)
                    loss_val = crit_(preds_, yb_int)
                    loss_val.backward()
                    optim_.step()

                    iteration_loss += loss_val.item()
                    train_correct += (torch.argmax(preds_, 1) == yb_int).sum().item()
                    train_total += yb_int.size(0)

            exec_time = time.time() - start_time
            train_accuracy = train_correct / (train_total+1e-9)
            print(f"Iteration {i}, YOPO training done. Loss={iteration_loss:.4f}, Acc={train_accuracy:.4f}, Time={exec_time:.2f}s")

            # 8) Evaluate if we have GT
            if args.gt_known and gt is not None:
                hom, com, v_m = homogeneity_completeness_v_measure(gt, labels_temp)
                print(f"Homo={hom:.4f}, Comp={com:.4f}, V-measure={v_m:.4f}")
                logfile.write(f"Iter={i}, K={K_est}, Homo={hom:.4f}, Comp={com:.4f}, V={v_m:.4f}\n")

                if args.true_k and (K_est == args.true_k):
                    aligned = align_cluster_index(gt, labels_temp)
                    accuracy_ = np.sum(aligned == gt) / len(gt)
                    print(f"Raw Accuracy with alignment={accuracy_:.4f}")
                    logfile.write(f"Raw Accuracy at iteration {i} = {accuracy_:.4f}\n")

            # 9) Save checkpoint
            checkpoint = {
                'model': model,
                'labels': labels,
                'DBI_best': DBI_best,
                'iteration': i,
                'done': done
            }
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{i}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            i += 1
            if i >= Config.M:
                done = True

        # End while
        print(f"Saving final model to {Config.model_path} and labels to {Config.label_path}")
        torch.save(model, Config.model_path)
        pickle_dump(labels, Config.label_path)


# ----------------------
# Additional align function
# ----------------------
def align_cluster_index(ref_cluster, map_cluster):
    """
    Re-map cluster indices in map_cluster so they match ref_cluster
    as best as possible, via Hungarian assignment.
    """
    ref_values = np.unique(ref_cluster)
    map_values = np.unique(map_cluster)
    if ref_values.shape[0] != map_values.shape[0]:
        print("Warning: align_cluster_index() with different #unique. Doing partial alignment.")
    cont_mat = contingency_matrix(ref_cluster, map_cluster)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-cont_mat)
    map_dict = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(ref_values) and c < len(map_values):
            ref_val = ref_values[r]
            map_val = map_values[c]
            map_dict[map_val] = ref_val

    map_cluster_aligned = np.array(map_cluster, copy=True)
    for k_ in map_dict:
        map_cluster_aligned[map_cluster == k_] = map_dict[k_]
    return map_cluster_aligned


if __name__ == "__main__":
    main()
