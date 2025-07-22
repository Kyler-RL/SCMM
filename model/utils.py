import sys
import logging
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from sklearn import preprocessing


# Generate soft assignments for soft contrastive learning.
def generate_soft_assignment(x, upper_bound, sharpness, metric="cosine"):
    x = x.reshape(x.shape[0], -1)
    x_expand = x.unsqueeze(0).expand(x.shape[0], x.shape[0], x.shape[1])
    if metric == "manhattan":
        dist_matrix = (x_expand - x_expand.permute(1, 0, 2)).abs().sum(dim=2)
    elif metric == "euclidean":
        dist_matrix = (x_expand - x_expand.permute(1, 0, 2)).pow(2).sum(dim=2).sqrt()
    elif metric == "cosine":
        dist_matrix = - F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
    else:
        raise ValueError("Invalid distance metric.")

    temp_dist_matrix = torch.triu(dist_matrix, diagonal=1)[:, 1:] + torch.tril(dist_matrix, diagonal=-1)[:, :-1]
    diagonal_indices = np.diag_indices(x.shape[0])
    dist_matrix[diagonal_indices] = temp_dist_matrix.min()
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    norm_dist_matrix = torch.Tensor(min_max_scaler.fit_transform(dist_matrix.detach().cpu().numpy())).to(x.device)
    soft_assignments = 2 * upper_bound / (1 + torch.exp(norm_dist_matrix / sharpness))

    return soft_assignments


# Construct matrix for soft labels.
def dup_matrix(soft_labels):
    mat_0 = torch.triu(soft_labels, diagonal=1)[:, 1:]
    mat_0 += torch.tril(soft_labels, diagonal=-1)[:, :-1]
    mat_1 = torch.cat([mat_0, soft_labels], dim=1)
    mat_2 = torch.cat([soft_labels, mat_0], dim=1)
    return mat_1, mat_2


# Use logger to print the output to the console and log files.
def _logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)

    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger
