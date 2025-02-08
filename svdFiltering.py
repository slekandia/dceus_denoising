import numpy as np
from copy import copy


def svdFiltering(data4d):

    # SVD variables
    svds = 9
    svdthresh = 4

    data = data4d["data4d"]
    filtered_data = np.empty_like(data, dtype=np.float64)

    for x in range(0, data.shape[0], svds):
        for y in range(0, data.shape[1], svds):
            for z in range(0, data.shape[2], svds):
                filtered_data[x:x + svds, y:y + svds, z:z + svds, :] = filter_svd(
                    data[x:x + svds, y:y + svds, z:z + svds, :], svdthresh)

    filtered_data4d = copy(data4d)
    filtered_data4d["data4d"] = filtered_data

    check_cartesian_voxel_data_validity(filtered_data4d, np.float_)

    return filtered_data4d


def filter_svd(s, thresh):
    # Reshape to NxT matrix with T samples of N pixels
    S = np.reshape(s, (s.shape[0] * s.shape[1] * s.shape[2], s.shape[3]))

    # Decompose: S = U @ diag(A) @ Vt   (Vt is the transpose of V)
    U, A, Vt = np.linalg.svd(S, full_matrices=False)

    # Filter smaller singular values (A is sorted in descending order)
    Af = A
    Af[thresh:] = 0

    # Reconstruct S based on filtered singular values
    Sf = np.dot(U * Af, Vt)

    return np.reshape(Sf, s.shape)
