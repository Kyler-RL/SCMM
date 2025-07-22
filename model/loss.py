import numpy as np
import torch
import torch.nn.functional as F
from utils import generate_soft_assignment
from torch import nn


# Adaptive automatic weighted multi-task loss function.
class AutoWeightLoss(nn.Module):
    def __init__(self, loss_num):
        super(AutoWeightLoss, self).__init__()
        params = torch.ones(loss_num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


# Generate boolean mask for positive and negative pairs.
def generate_pos_neg_mask(mask_num, sim_matrix, cur_batch_size):
    self_loop_mask = torch.eye(cur_batch_size, dtype=torch.bool)
    oral_batch_size = cur_batch_size // (mask_num + 1)
    pos_mask = np.zeros(sim_matrix.size())

    for i in range(mask_num + 1):
        pos_mask_right = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
        pos_mask_left = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
        pos_mask += pos_mask_right + pos_mask_left

    pos_mask = torch.from_numpy(pos_mask).to(sim_matrix.device)
    pos_mask[self_loop_mask] = 0
    neg_mask = 1 - pos_mask
    neg_mask[self_loop_mask] = 0

    return pos_mask.bool(), neg_mask.bool()


# Contrastive loss function with soft assignments for all samples except the original sample.
class SoftContrastiveLoss(nn.Module):
    def __init__(self, configs):
        super(SoftContrastiveLoss, self).__init__()
        self.mask_num = configs.mask_num
        self.dist_metric = configs.dist_metric
        self.upper_bound = configs.upper_bound
        self.sharpness = configs.sharpness
        self.temperature = configs.temperature

    def forward(self, batch_oral_mask_input, batch_oral_mask_emb):
        cur_batch_shape = batch_oral_mask_emb.shape
        # Compute similarity matrix.
        sim_matrix = torch.matmul(batch_oral_mask_emb, batch_oral_mask_emb.T)

        # Generate positive and negative masks of oral_samples and mask_samples.
        pos_mask, neg_mask = generate_pos_neg_mask(self.mask_num, sim_matrix, cur_batch_shape[0])
        # Locate positive pairs and negative pairs.
        positives = sim_matrix[pos_mask].view(cur_batch_shape[0], -1)
        negatives = sim_matrix[neg_mask].view(cur_batch_shape[0], -1)

        # Generate logits(similarity matrix) for positive and negative pairs.
        logits = torch.cat((positives, negatives), dim=-1)
        # Generate soft assignments based on the distance matrix in the original data space for sample pairs.
        soft_assignments = generate_soft_assignment(batch_oral_mask_input, self.upper_bound,
                                                    self.sharpness, metric=self.dist_metric)
        pos_assignments = torch.ones(positives.shape[0], positives.shape[1]).to(batch_oral_mask_emb.device).float()
        neg_soft_assignments = soft_assignments[neg_mask].view(cur_batch_shape[0], -1)
        soft_assignments = torch.cat((pos_assignments, neg_soft_assignments), dim=-1)

        # Soft contrastive loss.
        logits = -F.log_softmax(logits / self.temperature, dim=-1)
        loss_c = torch.sum(logits * soft_assignments) / (logits.shape[0] * logits.shape[1])

        return sim_matrix, loss_c


# Aggregation Reconstruction.
class AggregateReconstruct(nn.Module):
    def __init__(self, configs):
        super(AggregateReconstruct, self).__init__()
        self.temperature = configs.temperature
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, sim_matrix, batch_oral_mask_emb):
        cur_batch_shape = batch_oral_mask_emb.shape
        sim_matrix /= self.temperature
        sim_matrix -= torch.eye(cur_batch_shape[0]).to(sim_matrix.device).float() * 1e12
        reconstruct_weight_matrix = self.softmax(sim_matrix)

        # Generate the reconstructed batch embeddings.
        batch_oral_mask_emb = batch_oral_mask_emb.reshape(cur_batch_shape[0], -1)
        reconstruct_batch_emb = torch.matmul(reconstruct_weight_matrix, batch_oral_mask_emb)

        return reconstruct_weight_matrix, reconstruct_batch_emb


# Reconstruction loss for oral_samples and reconstruct_samples.
class ReconstructiveLoss(nn.Module):
    def __init__(self):
        super(ReconstructiveLoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x_in, reconstruct_x):
        x = x_in.reshape(x_in.shape[0], -1).detach()
        loss_r = self.mse(x, reconstruct_x)
        return loss_r
