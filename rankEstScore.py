import numpy as np
from tensor_operations import *


def rankEstScore(data4d, s):
    # Estimates the rank according to an information theoretic criteria
    # for tensors as described at Yokota, Tatsuya, Namgil Lee, and Andrzej
    # Cichocki. "Robust multilinear tensor rank estimation using higher
    # order singular value decomposition and information criteria."
    # IEEE Transactions on Signal Processing 65.5 (2016): 1196 - 1206.
    # Returns multilinear rank "rank_SCORE" that holds(R_1, R_2, ..., R_N)
    # data4d = data4d.astype(float)
    print('>> Perform Rank Estimation via SCORE Algorithm...')
    # rho_hat is suggested to be between 0.0001 and 0.01. It is responsible for describing
    # the threshold during the truncation. A smaller value is preferred for
    # robustness to noise.
    rho_hat = 10 ** - 5
    n = data4d.ndim
    rank_SCORE = np.zeros((n,), dtype='int')
    for i in range(n):
        mode_i_H = tens2mat(s, i)
        u_j = np.zeros(mode_i_H.shape[1])
        for h in range(mode_i_H.shape[1]):
            u_j[h] = np.dot(mode_i_H[:, h], mode_i_H[:, h])
        high_u_j_idx = np.argsort(u_j)[::-1]
        threshold = int(np.ceil(rho_hat * np.prod(data4d.shape) / data4d.shape[i]))
        # threshold = 5
        mode_i_H_P = mode_i_H[:, high_u_j_idx[:threshold]]
        tmp = 1 / threshold * np.dot(mode_i_H_P, mode_i_H_P.T)
        lambda_ = np.diag(tmp)
        lambda_ = np.sort(lambda_)[::-1]
        min_rank = data4d.shape[i]
        min_rank_val = 10 ** 6
        for r in range(1, data4d.shape[i]):
            denominator = 0
            nominator = 1
            for m in range(r, data4d.shape[i]):
                nominator *= lambda_[m] ** (1 / (data4d.shape[i] - r))
                denominator += 1 / (data4d.shape[i] - r) * lambda_[m]
            min_rank_val_new = -2 * np.log((nominator / denominator)**(threshold * (data4d.shape[i] - r))) + r * (2 * data4d.shape[i] - r) * np.log(threshold)
            if min_rank_val_new < min_rank_val:
                min_rank = r
                min_rank_val = min_rank_val_new
        rank_SCORE[i] = min_rank
    print(f"The estimated rank is {rank_SCORE}")
    return rank_SCORE
