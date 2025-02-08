import copy
from tensor_operations import *


def oitnnFiltering(data4d):
    """
    Taken from
    Andong WANG, Chao LI, Zhong JIN, Qibin ZHAO.
    "Robust Tensor Decomposition via Orientation Invariant Tubal Nuclear Norms," in AAAI 2020
    Args:
        data4d: the input tensor that holds the 4D DCEUS sequences at data4d["data4d"]

    Returns:

    """
    gammaL = 10000
    gammaO = 30

    data = data4d["data4d"]

    # # Observation
    obs = {}
    obs['tY'] = data
    # # Initialization of variables
    Y = data4d['data4d'].astype(float)
    # Algorithm parameters
    rho = 1
    nu = 1
    [_, alpha] = rolling_median(data)
    # Algorithm options
    optsO = {}
    optsO['lambdaL'] = gammaL
    optsO['lambdaS'] = gammaO
    # Maximum value is rolling median with a window of 5 in the time axis
    optsO['alpha'] = alpha
    optsO['rho'] = rho
    optsO['nu'] = nu
    optsO['MAX_ITER_OUT'] = 1000
    optsO['MAX_RHO'] = 1e10
    optsO['MAX_EPS'] = 5e-4
    optsO['verbose'] = 1

    # Construct memo
    memoO = h_construct_memo_v2(optsO)

    optsO['showImg'] = 1

    # Run
    memoO = f_rtd_OITNN_O(obs, optsO, memoO)

    Lhat = memoO['Lhat']
    del memoO["Lhat"]
    filtered_data4d = copy.copy(data4d)
    filtered_data4d["data4d"] = np.clip(Lhat, 0, 255)
    filtered_data4d["memo"] = memoO
    return filtered_data4d


def rolling_median(data):
    data_med = np.zeros(data.shape)
    for i in range(2, data.shape[-1] - 3):
        data_med[:, :, :, i] = np.median(data[:, :, :, i - 2:i + 3], axis=3)
    S = data - data_med
    alpha = np.max(data_med)
    return [S, alpha]


def h_construct_memo_v2(opts):
    max_iter = opts['MAX_ITER_OUT']
    memo = {}
    memo['max_iter'] = max_iter
    memo['opts'] = opts
    # relative error between every two iterations
    memo['eps'] = np.zeros(max_iter - 2)  # -2 due to iteration 0 and 1 give the relative error denominator

    return memo


def f_rtd_OITNN_O(obs, opts, memo):
    sz = obs['tY'].shape
    K = len(sz)
    gammaL = opts['lambdaL']
    alpha = opts['alpha']
    gammaO = opts['lambdaS']
    rho = opts['rho']
    nu = opts['nu']

    if 'vW' not in opts:
        weights = np.ones(K) / K
    else:
        weights = opts['vW']

    tY = obs['tY']
    tL = np.zeros(sz)

    tW = np.zeros(sz)
    tS = np.zeros(sz)
    tT = np.zeros(sz)
    tZ = np.zeros(sz)
    tK = np.zeros(sz)

    cK = [f_3DReshape(tW, k) for k in range(K)]
    cY = [f_3DReshape(tW, k) for k in range(K)]
    tmp = 1 + (1 + K) * (1 + rho)
    proxTNN_tau = gammaL * weights[0] / rho
    proxL1_tau = gammaO / rho
    #  plt.figure()
    for iter in range(0, opts['MAX_ITER_OUT']):
        tLold = tL.copy()

        sumK_ = np.zeros(sz)

        # Update L and S
        tT_ = tT + tZ / rho
        tK_ = tK + tW / rho

        for k in range(0, K):
            sumK_ = sumK_ + f_3DReshapeInverse(cK[k] + cY[k] / rho, sz, k)

        tS = (K + 1) * tY + (K + rho + K * rho) * tT_ - tK_ - sumK_
        tS /= tmp

        tL = (1 + rho) * tK_ + (1 + rho) * sumK_ + tY - tT_
        tL = np.clip(tL / tmp, 0, 255)
        # Update K_k, T, K
        for k in range(0, K):
            cK[k], _ = f_prox_TNN(f_3DReshape(tL, k) - cY[k] / rho, proxTNN_tau)
        [tT, _] = f_prox_l1(tS - tZ / rho, proxL1_tau)
        T_tmp = tL - tW / rho
        tK = np.sign(T_tmp) * np.minimum(np.abs(T_tmp), alpha)

        # Update Lagrangian parameters
        tW = tW + rho * (tK - tL)
        tZ = tZ + rho * (tT - tS)

        for k in range(0, K):
            cY[k] = cY[k] + rho * (cK[k] - f_3DReshape(tL, k))

        # Print iteration state and check convergence
        if iter <= 1:
            infE_zero = np.linalg.norm(tL - tLold)
        #  plt.plot(tL[30,30,30,:])
        memo['eps'][iter] = np.linalg.norm(tL - tLold) / infE_zero
        if memo['eps'][iter] < 0.1:  # 20 dB
            print("Convergence is reached")
            break
        rho = min(rho * nu, opts['MAX_RHO'])
    #  plt.savefig("tL.png")
    memo['Lhat'] = tL
    memo['Shat'] = tS

    return memo


def flatten_tuples(t):
    for x in t:
        if isinstance(x, tuple):
            yield from flatten_tuples(x)
        else:
            yield x


def f_3DReshapeInverse(X, sz, k):
    K = len(sz)
    idx_aug = list(range(0, K)) * 2
    R = idx_aug[k]
    vC = idx_aug[k + 2: k + K][::-1]
    t = idx_aug[k + 1]
    idx_p = [R] + list(vC) + [t]

    reshape_arr_ = (sz[R], tuple([(np.array(sz)[i]) for i in vC]), sz[t])
    reshape_arr = tuple(flatten_tuples(reshape_arr_))
    tmp = np.reshape(X, reshape_arr)
    idx_new = np.zeros(K, dtype=int)
    for kk in range(0, K):
        idx_new[kk] = np.where(np.array(idx_p) == kk)[0]

    tmp = np.transpose(tmp, idx_new)

    return tmp


def f_3DReshape(X, k):
    sz = X.shape
    K = len(sz)
    idx_aug = list(range(0, K)) * 2
    R = idx_aug[k]
    vC = idx_aug[k + 2: k + K][::-1]  # Reordering because of the C style vs Fortran style, here we do fortran
    t = idx_aug[k + 1]
    idx_p = [R] + vC + [t]
    tmp = np.transpose(X, idx_p)
    C = np.prod(np.array(sz)[vC])
    tmp = np.reshape(tmp, (sz[R], C, sz[t]))
    return tmp


def h_tnorm(X):
    return np.linalg.norm(X.flatten())


def f_prox_l1(Y, rho):
    x = np.sign(Y) * np.maximum(np.abs(Y) - rho, 0)
    f = np.sum(np.abs(x))
    return x, f


def f_tsvd_f(X):
    sz = X.shape
    X = np.fft.fft(X, axis=2)
    r = min(sz[0], sz[1])
    U = np.zeros((sz[0], r, sz[2]), dtype=np.complex128)
    S = np.zeros((r, r, sz[2]), dtype=np.complex128)
    V = np.zeros((r, sz[1], sz[2]), dtype=np.complex128)

    iMid = np.floor(sz[2] / 2).astype('int') + 1
    for i in range(0, iMid):
        U[:, :, i], S_tmp, V[:, :, i] = np.linalg.svd(X[:, :, i], full_matrices=False)
        S[:, :, i] = np.diag(S_tmp)
    for i in range(iMid, sz[2]):
        k = sz[2] - i
        U[:, :, i] = np.conj(U[:, :, k])
        S[:, :, i] = S[:, :, k]
        V[:, :, i] = np.conj(V[:, :, k])
    return U, S, V


def f_prox_TNN(Y, rho):
    n1, n2, n3 = Y.shape

    U, S, V = f_tsvd_f(Y)

    S, objV = f_prox_l1(S.real, rho)
    X = np.zeros((n1, n2, n3), dtype="complex128")
    mid = np.floor((n3 + 1) / 2).astype('int') + 1

    X[:, :, 0] = U[:, :, 0] @ S[:, :, 0] @ V[:, :, 0]
    for i in range(1, mid):
        X[:, :, i] = U[:, :, i] @ S[:, :, i] @ V[:, :, i]
        X[:, :, n3 - i] = np.conj(X[:, :, i])
    X = (np.fft.ifft(X, axis=2)).real

    return X, objV
