import copy
import numpy as np
from tensor_operations import *
from rankEstScore import rankEstScore


def mlsvdFiltering(data4d):
    """
    Computes the truncated Multilinear Singular Value Decomposition of a four-dimensional tensor along all four modes
    and sets it as the data4d of the input dictionary.

    Parameters:
        data4d: The dictionary that holds the resolution equalized ultrasound recording.

    Returns:
        data4d: The same dictionary with the data4d key updated with the truncated MLSVD.
    """
    data = data4d["data4d"].astype('float')
    u, s, sv = mlsvd(data)
    rank = rankEstScore(data, s)
    u_tr = []
    for i in range(4):
        u_tr.append(u[i][:, 0:rank[i]])
    s_tr = s[0:rank[0], 0:rank[1], 0:rank[2], 0:rank[3]]
    filtered_data4d = copy.copy(data4d)
    filtered_data4d["data4d"] = np.clip(lmlragen(u_tr, s_tr), 0, 255)
    return filtered_data4d
