import numpy as np

def reorder(indices, mode):
    """Reorders the elements
    """
    indices = list(indices)
    element = indices.pop(mode)
    return ([element] + indices[::-1])


def mat2tens(unfolded, shape, mode):
    """Returns the folded tensor of shape `shape` from the `mode`-mode unfolding `unfolded`.
    """
    unfolded_indices = reorder(range(len(shape)), mode)
    original_shape = [shape[i] for i in unfolded_indices]
    unfolded = unfolded.reshape(original_shape)

    folded_indices = list(range(len(shape) - 1, 0, -1))
    folded_indices.insert(mode, 0)
    return np.transpose(unfolded, folded_indices)


def tens2mat(tensor, mode):
    """
    Contracts a tens according to the n-th mode.

    Input: tens of size
           mode is the axis at which the tensor will be contracted
    Output: the tensor matrix product where the ith dimension is replaced by the row dimension of the matrix
    """

    d = tensor.shape
    nd = len(tensor.shape)
    assert mode < nd, "The mode should be less than the dimension of the tensor"

    row_d = d[mode]
    return np.transpose(tensor, reorder(range(tensor.ndim), mode)).reshape((row_d, -1))


def tmprod(tensor, mat, mode):
    """
    Computes the mode-n product of a tensor and a matrix.

    Input: Tensor an n-dimensional tensor
           mat a matrix
    Output: The resulting tensor matrix product.
    """
    if (1 in mat.shape):
        out_shape = list(tensor.shape)
        out_shape[mode] = 1
        result = np.zeros(out_shape)
        # Iterate over each mode-n slice and perform dot product with the vector
        for idx in range(tensor.shape[mode]):
            result = result + np.take(tensor, idx, mode) * mat[idx]
        return result
    else:
        out_n = np.matmul(mat, tens2mat(tensor, mode))
        out_shape = list(tensor.shape)
        out_shape[mode] = mat.shape[0]
        return mat2tens(out_n, out_shape, mode)


def khatri_rao(A, B):
    """
    Computes the column wise kronecker product

    """
    sz_A = np.shape(A)
    sz_B = np.shape(B)
    C = np.zeros((sz_A[0] * sz_B[0], sz_A[1]))
    for r in range(sz_A[1]):
        C[:, r] = np.kron(A[:, r], B[:, r])
    return C


def mlsvd(tensor, ranks=None):
    """
    Computes the multilinear singular value decomposition of a tensor and returns the core matrices and the factor
    matrices.

    Input: an N dimensional tensor
           args stand for a list with ranks for each N dimensions that will be used to truncate the tensor

    Output: the core matrices, a list of factor matrices, and the singular values in each unfolding
    """
    factors = []
    singular_values = []
    nd = len(tensor.shape)

    for n in range(nd):
        tensor_n = tens2mat(tensor, n)
        U, S, Vt = np.linalg.svd(tensor_n, full_matrices=False)
        if ranks is None:
            factors.append(U)
            singular_values.append(S)
        else:
            factors.append(U[:, 0:ranks[n]])
            singular_values.append(S[0:ranks[n]])
        if n == 0:
            core = tmprod(tensor, factors[n].conj().T, n)
        else:
            core = tmprod(core, factors[n].conj().T, n)
    return factors, core, singular_values


def tens2vec(tensor):
    """
    Flattens tensor to a vec
    """
    vec_indices = list(range(tensor.ndim - 1, -1, -1))
    return np.transpose(tensor, vec_indices).flatten()


def vec2tens(vec, shape):
    """
    Folds a vector to tensor
    """
    tens = vec.reshape(shape[::-1])
    return np.transpose(tens,list(range(len(shape) - 1, -1, -1)))


def lmlragen(U, S):
    """
    Returns the tensor T that multiplies each factor matrix in U in the corresponding mode with the core matrix S

    Input:
        U : List of the factor matrices
        S : The core tensor
    Output:
        T: The tensor that is the multiplication each factor matrix in U in the corresponding mode with
         the core matrix S
    """
    nd = len(U)
    for n in range(nd):
        if n == 0:
            T = tmprod(S, U[n], n)
        else:
            T = tmprod(T, U[n], n)
    return T


def generate(size_tens, rank_tens):
    """
    Returns a random tensor of size "size_tens" with the rank "rank_tens". The core and the factor matrices are sampled
    from the normal distribution with variance 1. In addition, the factor matrices are randomized according to the Haar
    measure.

    Input:
        size_tens: the size of the tensor
        rank_tens: the n-rank of the tensor
    Output:
        the random tensor of size "size_tens" with the rank "rank_tens"
    """
    s = np.random.normal(0, 1, size=rank_tens)
    u_list = []
    for i in range(len(size_tens)):
        u = np.random.normal(0, 1, size=[size_tens[i],
                                         rank_tens[i]])
        # Haar measure
        q, r = np.linalg.qr(u)
        u = q @ np.diag(np.sign(np.diag(r)))
        u_list.append(u)
    return lmlragen(u_list, s)


def generate_core_fac(size_tens, rank_tens):
    """
    Returns a random tensor of size "size_tens" with the rank "rank_tens". The core and the factor matrices are sampled
    from the normal distribution with variance 1. In addition, the factor matrices are randomized according to the Haar
    measure.

    Input:
        size_tens: the size of the tensor
        rank_tens: the n-rank of the tensor
    Output:
        the random tensor of size "size_tens" with the rank "rank_tens"
    """
    s = np.random.normal(0, 1, size=rank_tens)
    u_list = []
    for i in range(len(size_tens)):
        u = np.random.normal(0, 1, size=[size_tens[i],
                                         rank_tens[i]])
        # Haar measure
        q, r = np.linalg.qr(u)
        u = q @ np.diag(np.sign(np.diag(r)))
        u_list.append(u)
    return u_list, s


def tSVD(tensor):
    """
    Returns the tSVD decomposition
    Input:
        tensor: the tensor to be decomposed
    Output:
        [U, S, V]: the tensors that decompose the tSVD
    """
    sz = list(tensor.shape)
    szU = sz.copy()
    szU[1] = sz[0]
    szV = sz.copy()
    szV[0] = sz[1]

    U = np.zeros(szU,dtype="complex_")
    V = np.zeros(szV,dtype="complex_")
    S = np.zeros(sz,dtype="complex_")
    N = len(tensor.shape)
    tensor_fft = tensor.copy()
    for i in range(N - 1, 1, -1):
        tensor_fft = np.fft.fft(tensor_fft, axis=i)
    for index in np.ndindex(*sz[2:]):
        idx = (slice(None),slice(None)) + index
        U[idx], Stmp, Vtmp = np.linalg.svd(tensor_fft[idx])
        S[idx] = np.diag(Stmp)
        Vtmp = np.conj(Vtmp.T)
        V[idx] = Vtmp
    for i in range(N - 1, 1, -1):
        U = np.fft.ifft(U, axis=i)
        V = np.fft.ifft(V, axis=i)
        S = np.fft.ifft(S, axis=i)
    return [U.real, S.real, V.real]


def tprod(tensor1, tensor2):
    """
    Returns the t-product between two tensors
    """
    N = len(tensor1.shape)
    sz = tensor1.shape
    tensor1_fft = tensor1.copy()
    tensor2_fft = tensor2.copy()
    tensor_ret = np.zeros((tensor1.shape[0],tensor2.shape[1],*sz[2:]),dtype="complex_")
    for i in range(N - 1, 1, -1):
        tensor1_fft = np.fft.fft(tensor1_fft, axis=i)
        tensor2_fft = np.fft.fft(tensor2_fft, axis=i)
    for index in np.ndindex(*sz[2:]):
        idx = (slice(None), slice(None)) + index
        tensor_ret[idx] = np.matmul(tensor1_fft[idx], tensor2_fft[idx])
    for i in range(N - 1, 1, -1):
        tensor_ret = np.fft.ifft(tensor_ret, axis=i)
    return tensor_ret.real


def ttranspose(tensor):
    """
    Recursively reorder the tensor by reversing slices along higher dimensions,
    but keeping the first index (0th slice) fixed. Then transpose the first two dimensions.

    Parameters:
    tensor (numpy.ndarray): A tensor of shape (I1, I2, I3, ..., IN)
    axis (int): The current axis along which to reverse (starts from the last axis and moves backwards).

    Returns:
    numpy.ndarray: The T-transposed tensor with reversed slices along higher dimensions.
    """
    # Initialize axis to start from the last dimension
    idx = []
    N = len(tensor.shape)
    sz = list(tensor.shape)
    sz[0] = tensor.shape[1]
    sz[1] = tensor.shape[0]
    ttensor = np.zeros(sz)

    for i in range(2, N):
        idx.append(np.insert(np.arange(start=tensor.shape[i] - 1, stop=0, step=-1), 0, 0))

    grid = np.array(np.meshgrid(*idx,indexing='ij')).T.reshape(-1, len(idx))
    for count, ndidx in enumerate(grid):
        print(ndidx)
        j = (slice(None), slice(None)) + tuple(ndidx)
        orig_j = (slice(None), slice(None)) + np.unravel_index(count, sz[2:], order='F')
        ttensor[orig_j] = tensor[j].T
    return ttensor


def tlmlragen(U, S, V):
    szU = U.shape
    szS = S.shape
    szV = V.shape
    tens = np.zeros(szS)
    for i in range(min(szU[0], szV[0])):
        u_slice = np.expand_dims(U[:, i, :],axis=1)
        s_slice = np.expand_dims(S[i, i, :],axis=(0,1))
        v_slice = np.expand_dims(V[:, i, :],axis=1)
        tens = tens + tprod(u_slice, tprod(s_slice, ttranspose(v_slice)))
    return tens

