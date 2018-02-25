import numpy as np
from copy import deepcopy
import numba as nb

@nb.jit  
def compute_cocluster_mat(labels):
    mat = np.zeros([labels.shape[0], labels.shape[0]])
    for i in range(labels.shape[0]):
        for j in range(i, labels.shape[0]):
            mat[i, j] = int(labels[i] == labels[j])
            mat[j, i] = mat[i, j]
    return mat

@nb.jit  
def compute_FMscores_local(lab1, lab2):
    mat1 = compute_cocluster_mat(lab1)
    mat2 = compute_cocluster_mat(lab2)
    
    FP_arr = np.sum((mat1 - mat2 == -1), axis = 1)
    FN_arr = np.sum((mat1 - mat2 == 1), axis = 1)
    TP_arr = np.sum((mat1 + mat2 == 2), axis = 1) - 1

    FMI_arr = np.zeros(TP_arr.shape)
    for i in range(len(TP_arr)):
        if TP_arr[i] != 0:
            FMI_arr[i] = float(TP_arr[i] / float(np.sqrt((TP_arr[i] + FP_arr[i]) * (TP_arr[i] + FN_arr[i]))))
        else:
            FMI_arr[i] = 0

    return FMI_arr