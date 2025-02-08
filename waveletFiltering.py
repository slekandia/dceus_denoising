import numpy as np
from scipy.signal import medfilt
import pywt
import copy


def barroisFiltering(data4d):
    """
    This is a function that implements the "Denoising of Contrast-Enhanced
    Ultrasound Cine Sequences Based on a Multiplicative Model" paper with the
    DOI: 10.1109/TBME.2015.2407835.
    Author: Metin Calis
    Date: 30/06/23
    Inputs:
    - data4d: a dictionary that holds data4d["data4d"]: N_x, N_y, N_z, N_t tensor

    Outputs:
    - denoised_data_4d which is the denoised tensor of the same size.
    """

    def fast_robust_cleaner(data4d, p, med_filt_length, wave_name):
        robust_clean_4d = data4d.copy()
        N_x, N_y, N_z, N_t = data4d.shape
        N_dec = 1  # first decomposition level is recommended for outlier rejection
        detail_coeffs = [None] * N_dec
        len_c_a = np.zeros(N_dec + 1).astype('int')
        for i in range(N_x):
            for j in range(N_y):
                for k in range(N_z):
                    robust_c_a = np.copy(robust_clean_4d[i, j, k, :])
                    smooth_a_k = medfilt(robust_c_a, med_filt_length)
                    residual_a_k = robust_c_a - smooth_a_k
                    lambda_k_idx = int(np.floor(len(residual_a_k) * p))  # most of the robust residuals, R_k, should be zero
                    tmp = np.sort(np.abs(residual_a_k))[::-1]
                    lambda_k = tmp[lambda_k_idx]
                    R_k = np.zeros(len(residual_a_k))
                    idx_outlier = np.where(np.abs(residual_a_k) > lambda_k)[0]
                    R_k[idx_outlier] = np.sign(residual_a_k[idx_outlier]) * (np.abs(residual_a_k[idx_outlier]) - lambda_k)
                    robust_c_a = robust_c_a - R_k
                    for level_forward in range(N_dec):
                        robust_c_a, c_D = pywt.dwt(robust_c_a, wave_name)
                        smooth_a_k = medfilt(robust_c_a, med_filt_length)
                        residual_a_k = robust_c_a - smooth_a_k
                        lambda_k_idx = int(np.floor(len(residual_a_k) * p))
                        tmp = np.sort(np.abs(residual_a_k))[::-1]
                        lambda_k = tmp[lambda_k_idx]
                        R_k = np.zeros(len(residual_a_k))
                        idx_outlier = np.where(np.abs(residual_a_k) > lambda_k)[0]
                        R_k[idx_outlier] = np.sign(residual_a_k[idx_outlier]) * (np.abs(residual_a_k[idx_outlier]) - lambda_k)
                        robust_c_a = robust_c_a - R_k
                        detail_coeffs[level_forward] = c_D
                        len_c_a[level_forward + 1] = len(robust_c_a)
                    len_c_a[0] = N_t
                    for level_backward in range(N_dec, 0, -1):
                        robust_c_a = pywt.idwt(robust_c_a, detail_coeffs[level_backward - 1], wave_name)[0:len_c_a[level_backward - 1]]
                    robust_clean_4d[i, j, k, :] = robust_c_a
        return robust_clean_4d

    def wavelet_denoising(smooth_data4d, wave_name):
        wavelet_denoised = np.zeros_like(smooth_data4d)
        N_x, N_y, N_z, N_t = smooth_data4d.shape
        N_dec = int(np.log2(N_t))  # decomposition level
        d_soft = [None] * N_dec
        len_c_a = np.zeros(N_dec + 1).astype('int')
        for i in range(N_x):
            for j in range(N_y):
                for k in range(N_z):
                    c_A = np.copy(smooth_data4d[i, j, k, :])
                    for level_k in range(N_dec):
                        c_A, c_D = pywt.dwt(c_A, wave_name)
                        if level_k == 0:
                            # Threshold selection
                            sigma = np.median(np.abs(c_D))
                            T = sigma * np.sqrt(2 * np.log(N_t))
                        # Soft threshold
                        d_soft[level_k] = pywt.threshold(c_D, T, mode='soft')
                        len_c_a[level_k + 1] = len(c_A)
                    len_c_a[0] = N_t
                    for level_k in range(N_dec, 0, -1):
                        c_A = pywt.idwt(c_A, d_soft[level_k - 1], wave_name)[0:len_c_a[level_k - 1]]
                    wavelet_denoised[i, j, k, :] = c_A
        return wavelet_denoised

    wave_name = 'bior3.9'
    data = data4d['data4d'].astype('float')
    data = np.clip(data - np.repeat(np.median(data[:, :, :, 0:12], axis=3)[:, :, :, np.newaxis], data.shape[3], axis=3),
                   0, 255)
    smooth_data4d = fast_robust_cleaner(data.astype('float'), 0.07, 7, wave_name)
    clean_data4d = np.clip(wavelet_denoising(smooth_data4d, wave_name), 0, 255)
    denoised_data4d = copy.copy(data4d)
    denoised_data4d["data4d"] = clean_data4d.astype('float')
    return denoised_data4d
