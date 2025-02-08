import copy
from tensor_operations import *
from rankEstScore import rankEstScore


def generalFiltering(data4d):
    """
    This function estimates the denoised four-dimensional tensor denoised_data4d
    assuming that the received signal follows a log-Rayleigh distribution.
    The method is based on the paper "An Optimal Statistical and Computational Framework for Generalized Tensor Estimation"
    with the DOI: 10.1214/21-AOS2061. The method is updated such that the log-Rayleigh loss function is used.

    Inputs:
    - data4d: a dictionary that has data4d['data4d'], the noisy ultrasound recording.

    Outputs:
    - denoised_data4d: a dictionary that has the same fields as the input struct and the denoised 4D tensor.
    """
    iter_converging = False
    eta = 10 ** -7
    DR = float(data4d['scan_settings']['dynamic_range_db'])
    log_c = 255 * np.log10(np.exp(1)) * 10 / DR
    # Initialization of variables
    Y = data4d['data4d'].astype(float)
    sz = Y.shape
    # Q0 = np.median(Y[:, :, :, 0:12], axis=3)
    # Y = np.clip(Y - np.repeat(Q0[:, :, :, np.newaxis], sz[3], axis=3), 0, 255)
    # mean_noise = 0.0579 * log_c this is according to the rayleigh noise assumption
    u, s, sv = mlsvd(Y)
    ranks = rankEstScore(Y, s)
    stopping_eps = 10 ** -1
    it = 0
    total_iter = 2 * 10 ** 4
    while not iter_converging:
        u_tr = []
        for i in range(4):
            u_tr.append(u[i][:, 0:ranks[i]])
        s_tr = s[0:ranks[0], 0:ranks[1], 0:ranks[2], 0:ranks[3]]
        X = np.clip(lmlragen(u_tr, s_tr), 0, 255)
        # Initialization of parameters
        lambda_ = 1e-3
        for i in range(len(sz)):
            moden = tens2mat(X, i)
            tmp = np.linalg.norm(moden)
            if tmp > lambda_:
                lambda_ = tmp
        a = lambda_
        b = lambda_ ** (1 / 4)
        for i in range(len(sz)):
            u_tr[i] = u_tr[i] * b
        s_tr = s_tr / a
        loss = np.zeros(total_iter)
        u_tr_t = [np.empty_like(u_tr[i].T) for i in range(4)]
        # Iteration
        for it in range(total_iter):
            X_old = X.copy()
            # Precalculate the transpose of U_tr
            u_tr_t[0] = u_tr[0].T
            u_tr_t[1] = u_tr[1].T
            u_tr_t[2] = u_tr[2].T
            u_tr_t[3] = u_tr[3].T
            # Calculate the gradients
            grad_l = (2 - np.exp(2 / log_c * (Y - X))) / log_c
            grad_u_tr_0 = tens2mat(tmprod(tmprod(tmprod(grad_l, u_tr_t[3], 3), u_tr_t[2], 2), u_tr_t[1], 1), 0) @ tens2mat(s_tr, 0).T + a * u_tr[0] @ (u_tr_t[0] @ u_tr[0] - b ** 2 * np.eye(ranks[0]))
            grad_u_tr_1 = tens2mat(tmprod(tmprod(tmprod(grad_l, u_tr_t[3], 3), u_tr_t[2], 2), u_tr_t[0], 0), 1) @ tens2mat(s_tr, 1).T + a * u_tr[1] @ (u_tr_t[1] @ u_tr[1] - b ** 2 * np.eye(ranks[1]))
            grad_u_tr_2 = tens2mat(tmprod(tmprod(tmprod(grad_l, u_tr_t[3], 3), u_tr_t[1], 1), u_tr_t[0], 0), 2) @ tens2mat(s_tr, 2).T + a * u_tr[2] @ (u_tr_t[2] @ u_tr[2] - b ** 2 * np.eye(ranks[2]))
            grad_u_tr_3 = tens2mat(tmprod(tmprod(tmprod(grad_l, u_tr_t[2], 2), u_tr_t[1], 1), u_tr_t[0], 0), 3) @ tens2mat(s_tr, 3).T + a * u_tr[3] @ (u_tr_t[3] @ u_tr[3] - b ** 2 * np.eye(ranks[3]))
            grad_s = lmlragen(u_tr_t, grad_l)
            # Gradient descent
            u_tr[0] = u_tr[0] - eta * grad_u_tr_0
            u_tr[1] = u_tr[1] - eta * grad_u_tr_1
            u_tr[2] = u_tr[2] - eta * grad_u_tr_2
            u_tr[3] = u_tr[3] - eta * grad_u_tr_3
            s_tr = s_tr - eta * grad_s
            # Form the low rank tensor
            X = np.clip(lmlragen(u_tr, s_tr), 0, 255)
            if it <= 1:
                loss_zero = np.linalg.norm(X - X_old)
            loss[it] = np.linalg.norm(X - X_old) / loss_zero
            print(loss[it])
            if loss[it] < stopping_eps:
                iter_converging = True
                print("STOPPING CONDITION REACHED FOR GENERAL FILTERING")
                break
            if(it > 1 and loss[it] > loss[it - 1]) or np.any(np.isnan(loss[it])):
                eta = eta / 10
                print('Diverging iteration, reduce the step size to ' + str(eta))
                break
        if it == total_iter - 1:
            break
    denoised_data4d = copy.copy(data4d)
    denoised_data4d["data4d"] = np.clip(X, 0, 255)
    denoised_data4d["loss"] = loss
    denoised_data4d["lr"] = eta
    return denoised_data4d
