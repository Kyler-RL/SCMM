import numpy as np
import math
import torch


# Random Masking.
def binomial_random_mask(B, C, F, mask_ratio):
    mask_id = np.random.binomial(1, 1 - mask_ratio, (B, C, F))
    return mask_id


# Channel Masking.
def binomial_channel_mask(B, C, F, mask_ratio):
    mask_id = np.ones((B, C, F), dtype=int)
    binomial_mask = np.random.binomial(1, 1 - mask_ratio, (B, C))
    indices = np.where(binomial_mask == 0)
    mask_id[indices[0], indices[1], :] = 0
    return mask_id


# Randomly generate mask sequence using binomial distribution.
def generate_binomial_mask(B, C, F, mask_ratio, threshold, mask_mode):
    if mask_mode == "random_mask":
        mask_id = binomial_random_mask(B, C, F, mask_ratio)
    elif mask_mode == "channel_mask":
        mask_id = binomial_channel_mask(B, C, F, mask_ratio)
    elif mask_mode == "parallel_mask":
        mask_id = np.zeros((B, C, F), dtype=int)
        random_mask_id = binomial_random_mask(B, C, F, mask_ratio)
        channel_mask_id = binomial_channel_mask(B, C, F, mask_ratio)

        # Parallel masking for each sample by uniform distribution.
        flag = np.random.uniform(0, 1, B) > threshold
        random_mask_pos, channel_mask_pos = np.where(flag), np.where(~flag)
        mask_id[random_mask_pos] = random_mask_id[random_mask_pos]
        mask_id[channel_mask_pos] = channel_mask_id[channel_mask_pos]
    elif mask_mode == "hybrid_mask":
        mask_id = np.zeros((B, C, F), dtype=int)
        random_mask_id = binomial_random_mask(B, C, F, mask_ratio)
        channel_mask_id = binomial_channel_mask(B, C, F, mask_ratio)

        # Hybrid masking fusion by uniform distribution.
        flag = np.random.uniform(0, 1, (B, C)) > threshold
        random_mask_pos, channel_mask_pos = np.where(flag), np.where(~flag)
        mask_id[random_mask_pos] = random_mask_id[random_mask_pos]
        mask_id[channel_mask_pos] = channel_mask_id[channel_mask_pos]
    else:
        raise ValueError(f'Error argument \'{mask_mode}\', please re-enter the correct mask_mode!')
    return torch.from_numpy(mask_id).to(torch.bool)


# Generate multiple(mask_num) masked sequences using binomial_mask strategy.
def masking(x, mask_ratio=None, mask_num=None, threshold=None, mask_mode="hybrid_mask"):
    # [batch_size, in_channels, fea_length]
    B, C, F = x.shape
    if mask_num is None:
        mask_num = math.ceil(1.5 / (1 - mask_ratio))

    # Mask operation.
    x_mask = torch.Tensor(np.tile(x, reps=(mask_num, 1, 1)))  # Repeat "x" for mask_num times.
    if mask_mode == "channel_mask" and C == 1:
        # [mask_num * batch_size, 1, 310] → [mask_num * batch_size, 62, 5]
        mask_id = generate_binomial_mask(x_mask.shape[0], 62, 5, mask_ratio=mask_ratio, threshold=threshold,
                                         mask_mode=mask_mode).reshape(mask_num * B, C, F).to(x.device)
    else:
        # [mask_num * batch_size, in_channels, fea_length]
        mask_id = generate_binomial_mask(x_mask.shape[0], C, F, mask_ratio=mask_ratio, threshold=threshold,
                                         mask_mode=mask_mode).to(x.device)

    x_mask[~mask_id] = 0
    return mask_id, x_mask
